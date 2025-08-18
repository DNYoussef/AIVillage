#!/usr/bin/env python3
"""AI Research Papers RAG Ingestion Script.

Ingests all AI research papers including:
- 147 papers from the ai_papers zip file
- Grossman Non-Newtonian Calculus PDF
- Grossman Meta-Calculus PDF

Processes PDFs with intelligent chunking and adds to RAG system.
"""

import asyncio
import hashlib
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import fitz  # PyMuPDF - more robust PDF processing
    import PyPDF2
except ImportError:
    print("Installing required PDF processing libraries...")
    os.system("pip install PyPDF2 PyMuPDF")
    import fitz
    import PyPDF2

from production.rag.rag_system.core.intelligent_chunking import DocumentType, IntelligentChunker
from production.rag.wikipedia_data_loader import WikipediaDataLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("rag_ingestion.log"),
    ],
)
logger = logging.getLogger(__name__)


class PDFIngestionManager:
    """Manages PDF ingestion into RAG system with intelligent chunking."""

    def __init__(self) -> None:
        self.loader = WikipediaDataLoader()
        self.chunker = IntelligentChunker(
            window_size=3,
            min_chunk_sentences=2,
            max_chunk_sentences=15,
            context_overlap=1,
        )
        self.processed_files = set()
        self.stats = {
            "total_files": 0,
            "successful_ingestions": 0,
            "failed_ingestions": 0,
            "total_chunks": 0,
            "total_pages": 0,
        }

    def extract_text_from_pdf(self, pdf_path: str) -> tuple[str, dict]:
        """Extract text from PDF using PyMuPDF (more robust than PyPDF2)."""
        try:
            doc = fitz.open(pdf_path)
            text_blocks = []
            metadata = {
                "total_pages": len(doc),
                "title": doc.metadata.get("title", ""),
                "author": doc.metadata.get("author", ""),
                "subject": doc.metadata.get("subject", ""),
                "creator": doc.metadata.get("creator", ""),
                "pdf_path": pdf_path,
            }

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    text_blocks.append(f"[Page {page_num + 1}]\n{text}")

            full_text = "\n\n".join(text_blocks)
            doc.close()

            if not full_text.strip():
                msg = "No text extracted from PDF"
                raise ValueError(msg)

            return full_text, metadata

        except Exception as e:
            logger.exception(f"Failed to extract text from {pdf_path}: {e}")
            # Fallback to PyPDF2
            try:
                with open(pdf_path, "rb") as file:
                    reader = PyPDF2.PdfReader(file)
                    text_blocks = []
                    metadata = {
                        "total_pages": len(reader.pages),
                        "title": getattr(reader.metadata, "title", "") if reader.metadata else "",
                        "author": getattr(reader.metadata, "author", "") if reader.metadata else "",
                        "pdf_path": pdf_path,
                    }

                    for page_num, page in enumerate(reader.pages):
                        text = page.extract_text()
                        if text.strip():
                            text_blocks.append(f"[Page {page_num + 1}]\n{text}")

                    return "\n\n".join(text_blocks), metadata

            except Exception as e2:
                logger.exception(f"Both PDF extraction methods failed for {pdf_path}: {e2}")
                raise

    def determine_document_type(self, text: str, filename: str) -> DocumentType:
        """Determine document type based on content and filename."""
        filename_lower = filename.lower()
        text_sample = text[:2000].lower()

        # Check for academic paper indicators
        academic_indicators = [
            "abstract",
            "introduction",
            "methodology",
            "results",
            "conclusion",
            "references",
            "arxiv",
            "doi:",
            "proceedings",
        ]

        # Check for technical indicators
        tech_indicators = [
            "algorithm",
            "implementation",
            "system",
            "framework",
            "architecture",
            "performance",
            "evaluation",
        ]

        if any(indicator in text_sample for indicator in academic_indicators):
            return DocumentType.ACADEMIC
        if any(indicator in text_sample for indicator in tech_indicators):
            return DocumentType.TECHNICAL
        if "calculus" in filename_lower or "mathematics" in filename_lower:
            return DocumentType.ACADEMIC
        return DocumentType.CONVERSATIONAL

    def create_file_hash(self, filepath: str) -> str:
        """Create unique hash for file to avoid duplicate processing."""
        with open(filepath, "rb") as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash

    async def ingest_pdf(self, pdf_path: str, category: str = "research") -> bool:
        """Ingest a single PDF into the RAG system."""
        try:
            logger.info(f"Processing PDF: {pdf_path}")

            # Check if already processed
            file_hash = self.create_file_hash(pdf_path)
            if file_hash in self.processed_files:
                logger.info(f"Skipping already processed file: {pdf_path}")
                return True

            # Extract text and metadata
            text, metadata = self.extract_text_from_pdf(pdf_path)

            if len(text.strip()) < 100:
                logger.warning(f"Insufficient text extracted from {pdf_path}")
                return False

            # Determine document type for optimal chunking
            doc_type = self.determine_document_type(text, Path(pdf_path).name)

            # Generate unique document ID
            filename = Path(pdf_path).stem
            doc_id = f"{category}_{filename}_{file_hash[:8]}"

            # Chunk the document intelligently
            chunks = self.chunker.chunk_document(text=text, document_id=doc_id, doc_type=doc_type)

            logger.info(f"Created {len(chunks)} intelligent chunks for {filename}")

            # Store document in RAG system
            try:
                # Add main document
                await self.loader.add_document(
                    content=text,
                    filename=filename,
                    metadata={
                        **metadata,
                        "category": category,
                        "document_type": doc_type.value,
                        "chunk_count": len(chunks),
                        "file_hash": file_hash,
                        "processing_method": "intelligent_chunking",
                    },
                )

                # Store individual chunks with metadata
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{doc_id}_chunk_{i}"
                    await self.loader.add_document(
                        content=chunk.text,
                        filename=f"{filename}_chunk_{i}",
                        metadata={
                            "parent_document": doc_id,
                            "chunk_index": i,
                            "chunk_id": chunk_id,
                            "content_type": chunk.content_type.value,
                            "topic_coherence": chunk.topic_coherence,
                            "word_count": chunk.word_count,
                            "sentence_count": len(chunk.sentences),
                            "summary": chunk.summary,
                            "entities": chunk.entities or [],
                            "category": category,
                        },
                    )

                self.processed_files.add(file_hash)
                self.stats["successful_ingestions"] += 1
                self.stats["total_chunks"] += len(chunks)
                self.stats["total_pages"] += metadata.get("total_pages", 0)

                logger.info(f"Successfully ingested {filename} with {len(chunks)} chunks")
                return True

            except Exception as e:
                logger.exception(f"Failed to store document {filename} in RAG system: {e}")
                return False

        except Exception as e:
            logger.exception(f"Failed to ingest PDF {pdf_path}: {e}")
            self.stats["failed_ingestions"] += 1
            return False

    def find_all_pdfs(self, base_path: str) -> list[tuple[str, str]]:
        """Find all PDF files and categorize them."""
        pdfs = []
        base_path = Path(base_path)

        for pdf_path in base_path.rglob("*.pdf"):
            # Determine category based on path
            path_str = str(pdf_path)
            if "Agent Forge" in path_str:
                category = "agent_forge"
            elif "HypeRAG" in path_str or "HybridRAG" in path_str:
                category = "rag_research"
            elif "Compression" in path_str:
                category = "compression"
            elif "SAGE" in path_str:
                category = "sage_research"
            elif "KING" in path_str:
                category = "king_research"
            elif "MAGI" in path_str:
                category = "magi_research"
            elif "multiagent" in path_str:
                category = "multiagent"
            elif "Math" in path_str or "geometry" in path_str or "calculus" in path_str.lower():
                category = "mathematics"
            elif "Grossman" in path_str:
                category = "grossman_calculus"
            else:
                category = "general_research"

            pdfs.append((str(pdf_path), category))

        return pdfs

    async def ingest_all_papers(self) -> None:
        """Ingest all AI research papers into RAG system."""
        logger.info("Starting comprehensive AI research paper ingestion")

        # Find all PDFs to process
        pdf_sources = []

        # 1. Extracted AI papers
        ai_papers_path = "/tmp/ai_papers"
        if os.path.exists(ai_papers_path):
            pdf_sources.extend(self.find_all_pdfs(ai_papers_path))

        # 2. Grossman calculus papers
        grossman_papers = [
            (
                r"C:\Users\17175\Downloads\Grossman, Non-Newtonian Calculus (1) (1).pdf",
                "grossman_calculus",
            ),
            (
                r"C:\Users\17175\Downloads\Grossman, Meta-Calculus (1).pdf",
                "grossman_calculus",
            ),
        ]

        for paper_path, category in grossman_papers:
            if os.path.exists(paper_path):
                pdf_sources.append((paper_path, category))

        self.stats["total_files"] = len(pdf_sources)
        logger.info(f"Found {len(pdf_sources)} PDF files to process")

        # Process PDFs in batches to avoid overwhelming the system
        batch_size = 5
        for i in range(0, len(pdf_sources), batch_size):
            batch = pdf_sources[i : i + batch_size]
            logger.info(f"Processing batch {i // batch_size + 1}/{(len(pdf_sources) - 1) // batch_size + 1}")

            tasks = []
            for pdf_path, category in batch:
                tasks.append(self.ingest_pdf(pdf_path, category))

            # Process batch concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Log batch results
            successful = sum(1 for r in results if r is True)
            logger.info(f"Batch complete: {successful}/{len(batch)} successful")

            # Brief pause between batches
            await asyncio.sleep(2)

        # Final statistics
        logger.info("=== INGESTION COMPLETE ===")
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"Successful ingestions: {self.stats['successful_ingestions']}")
        logger.info(f"Failed ingestions: {self.stats['failed_ingestions']}")
        logger.info(f"Total chunks created: {self.stats['total_chunks']}")
        logger.info(f"Total pages processed: {self.stats['total_pages']}")
        logger.info(f"Success rate: {self.stats['successful_ingestions'] / self.stats['total_files'] * 100:.1f}%")


async def main() -> None:
    """Main ingestion function."""
    try:
        manager = PDFIngestionManager()
        await manager.ingest_all_papers()

        print("\n✅ AI Research Papers RAG Ingestion Complete!")
        print(f"Successfully processed {manager.stats['successful_ingestions']} papers")
        print(f"Created {manager.stats['total_chunks']} intelligent chunks")
        print(f"Processed {manager.stats['total_pages']} total pages")

        # Test a sample query
        print("\n🔍 Testing RAG system with sample query...")
        # You can add a test query here if needed

    except Exception as e:
        logger.exception(f"Ingestion failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
