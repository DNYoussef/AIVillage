#!/usr/bin/env python3
"""Process STEM Textbooks for RAG Integration.

This script extracts and processes STEM textbooks from zip files, chunks them,
and ingests them into both vector and graph RAG systems.
"""

import asyncio
import json
import logging
import os
import shutil
import time
import zipfile
from pathlib import Path
from typing import Any

# Import PDF processing
try:
    import pdfplumber
    import PyPDF2

    PDF_AVAILABLE = True
except ImportError:
    print("Installing PDF processing dependencies...")
    os.system("pip install PyPDF2 pdfplumber")
    try:
        import pdfplumber
        import PyPDF2

        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False

# Import text processing
try:
    import nltk
    from sentence_transformers import SentenceTransformer

    TEXT_PROCESSING_AVAILABLE = True
except ImportError:
    print("Installing text processing dependencies...")
    os.system("pip install sentence-transformers nltk")
    try:
        import nltk
        from sentence_transformers import SentenceTransformer

        TEXT_PROCESSING_AVAILABLE = True
    except ImportError:
        TEXT_PROCESSING_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("stem_textbook_processing.log"),
    ],
)
logger = logging.getLogger(__name__)


class STEMTextbookProcessor:
    """Processor for STEM textbooks."""

    def __init__(self, downloads_path: str = "C:/Users/17175/Downloads") -> None:
        self.downloads_path = Path(downloads_path)
        self.work_dir = Path("data/stem_textbooks_processed")
        self.work_dir.mkdir(exist_ok=True)

        # Initialize components
        if TEXT_PROCESSING_AVAILABLE:
            self.embeddings_model = SentenceTransformer("paraphrase-MiniLM-L3-v2")
            try:
                nltk.download("punkt", quiet=True)
                nltk.download("stopwords", quiet=True)
            except:
                pass

        self.processed_books = {}
        self.chunks = {}
        self.processing_stats = {
            "books_extracted": 0,
            "books_processed": 0,
            "chunks_created": 0,
            "total_pages": 0,
            "total_words": 0,
            "processing_time": 0,
        }

    def extract_zip_files(self):
        """Extract STEM textbook zip files."""
        logger.info("Extracting STEM textbook zip files...")

        zip_files = [
            "STEM textbooks-20250810T172949Z-1-001.zip",
            "STEM textbooks-20250810T172949Z-1-002.zip",
        ]

        extraction_dir = self.work_dir / "extracted"
        extraction_dir.mkdir(exist_ok=True)

        extracted_files = []

        for zip_name in zip_files:
            zip_path = self.downloads_path / zip_name
            if not zip_path.exists():
                logger.warning(f"Zip file not found: {zip_path}")
                continue

            logger.info(f"Extracting {zip_name}...")

            try:
                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    # Get list of files
                    file_list = zip_ref.namelist()
                    pdf_files = [f for f in file_list if f.lower().endswith(".pdf")]

                    logger.info(f"Found {len(pdf_files)} PDF files in {zip_name}")

                    for pdf_file in pdf_files:
                        try:
                            # Extract with safe filename
                            safe_name = self._create_safe_filename(pdf_file)
                            extract_path = extraction_dir / safe_name

                            with zip_ref.open(pdf_file) as source:
                                with open(extract_path, "wb") as target:
                                    shutil.copyfileobj(source, target)

                            extracted_files.append(extract_path)

                        except Exception as e:
                            logger.warning(f"Failed to extract {pdf_file}: {e}")
                            continue

                self.processing_stats["books_extracted"] += len(
                    [f for f in file_list if f.lower().endswith(".pdf")]
                )

            except Exception as e:
                logger.exception(f"Failed to extract {zip_name}: {e}")
                continue

        logger.info(f"Extracted {len(extracted_files)} PDF files successfully")
        return extracted_files

    def _create_safe_filename(self, original_path: str) -> str:
        """Create a safe filename from the original path."""
        # Remove problematic characters and create a safe filename
        filename = Path(original_path).name
        # Replace problematic characters
        safe_chars = (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. "
        )
        safe_filename = "".join(c if c in safe_chars else "_" for c in filename)
        # Remove multiple underscores
        while "__" in safe_filename:
            safe_filename = safe_filename.replace("__", "_")

        return safe_filename[:100] + ".pdf"  # Limit length

    def extract_text_from_pdf(self, pdf_path: Path) -> dict[str, Any]:
        """Extract text from PDF file."""
        if not PDF_AVAILABLE:
            logger.error("PDF processing libraries not available")
            return {"text": "", "pages": 0, "error": "PDF libraries not available"}

        text_content = []
        page_count = 0

        try:
            # Try pdfplumber first (better text extraction)
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text:
                            text_content.append(text)
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract page {page_num} from {pdf_path.name}: {e}"
                        )
                        continue

        except Exception as e:
            logger.warning(f"pdfplumber failed for {pdf_path.name}, trying PyPDF2: {e}")

            # Fallback to PyPDF2
            try:
                with open(pdf_path, "rb") as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    page_count = len(pdf_reader.pages)

                    for page_num in range(page_count):
                        try:
                            page = pdf_reader.pages[page_num]
                            text = page.extract_text()
                            if text:
                                text_content.append(text)
                        except Exception as e:
                            logger.warning(
                                f"Failed to extract page {page_num} from {pdf_path.name}: {e}"
                            )
                            continue

            except Exception as e:
                logger.exception(f"Failed to extract text from {pdf_path.name}: {e}")
                return {"text": "", "pages": 0, "error": str(e)}

        full_text = "\n\n".join(text_content)
        word_count = len(full_text.split())

        return {
            "text": full_text,
            "pages": page_count,
            "word_count": word_count,
            "character_count": len(full_text),
        }

    def classify_textbook_subject(self, text: str, filename: str) -> str:
        """Classify textbook subject area."""
        filename_lower = filename.lower()
        text_sample = text[:2000].lower()  # First 2000 characters

        # Subject classification based on keywords
        subjects = {
            "mathematics": [
                "calculus",
                "algebra",
                "geometry",
                "statistics",
                "probability",
                "math",
                "theorem",
                "equation",
            ],
            "physics": [
                "physics",
                "quantum",
                "mechanics",
                "electromagnetism",
                "thermodynamics",
                "relativity",
                "particle",
            ],
            "chemistry": [
                "chemistry",
                "molecular",
                "organic",
                "inorganic",
                "reaction",
                "compound",
                "element",
            ],
            "biology": [
                "biology",
                "genetics",
                "molecular biology",
                "biochemistry",
                "cell",
                "organism",
                "evolution",
            ],
            "engineering": [
                "engineering",
                "design",
                "mechanical",
                "electrical",
                "civil",
                "chemical engineering",
            ],
            "computer_science": [
                "computer",
                "programming",
                "algorithm",
                "software",
                "machine learning",
                "data structure",
            ],
            "nanophysics": [
                "nano",
                "nanoscale",
                "nanotechnology",
                "nanoparticle",
                "nanomaterial",
            ],
            "economics": [
                "economics",
                "financial",
                "market",
                "economic",
                "finance",
                "business",
                "management",
            ],
        }

        # Score each subject
        scores = {}
        for subject, keywords in subjects.items():
            score = 0
            for keyword in keywords:
                if keyword in filename_lower:
                    score += 3  # Higher weight for filename
                if keyword in text_sample:
                    score += 1
            scores[subject] = score

        # Return highest scoring subject
        if scores:
            return max(scores, key=scores.get)
        return "general"

    def chunk_textbook_intelligently(
        self, text: str, doc_id: str, subject: str
    ) -> list[dict[str, Any]]:
        """Chunk textbook using intelligent chunking."""
        if not TEXT_PROCESSING_AVAILABLE:
            # Fallback to simple chunking
            return self._simple_chunk(text, doc_id)

        # Textbook-specific chunking parameters
        chunk_params = {
            "mathematics": {"size": 1500, "overlap": 200},
            "physics": {"size": 2000, "overlap": 300},
            "chemistry": {"size": 1800, "overlap": 250},
            "engineering": {"size": 2500, "overlap": 300},
            "computer_science": {"size": 2000, "overlap": 200},
        }

        params = chunk_params.get(subject, {"size": 2000, "overlap": 200})

        # Split into sentences
        try:
            sentences = nltk.sent_tokenize(text)
        except:
            # Fallback sentence splitting
            sentences = [s.strip() for s in text.split(".") if s.strip()]

        chunks = []
        current_chunk = []
        current_size = 0

        for _i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_size = len(sentence)

            # Start new chunk if current is too large
            if current_size + sentence_size > params["size"] and current_chunk:
                # Create chunk
                chunk_text = " ".join(current_chunk)
                chunk_id = f"{doc_id}_chunk_{len(chunks)}"

                # Extract keywords and entities
                keywords = self._extract_keywords(chunk_text, subject)
                entities = self._extract_entities(chunk_text)

                chunk_data = {
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "text": chunk_text,
                    "position": len(chunks),
                    "start_idx": 0,  # Could be calculated if needed
                    "end_idx": len(chunk_text),
                    "subject": subject,
                    "keywords": keywords,
                    "entities": entities,
                    "word_count": len(chunk_text.split()),
                    "coherence_score": self._calculate_coherence(chunk_text),
                    "trust_score": self._calculate_textbook_trust(doc_id, subject),
                }

                if TEXT_PROCESSING_AVAILABLE:
                    chunk_data["embedding"] = self.embeddings_model.encode(chunk_text)

                chunks.append(chunk_data)

                # Start new chunk with overlap
                overlap_sentences = (
                    current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                )
                current_chunk = overlap_sentences
                current_size = sum(len(s) for s in current_chunk)

            current_chunk.append(sentence)
            current_size += sentence_size

        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk_id = f"{doc_id}_chunk_{len(chunks)}"

            keywords = self._extract_keywords(chunk_text, subject)
            entities = self._extract_entities(chunk_text)

            chunk_data = {
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "text": chunk_text,
                "position": len(chunks),
                "start_idx": 0,
                "end_idx": len(chunk_text),
                "subject": subject,
                "keywords": keywords,
                "entities": entities,
                "word_count": len(chunk_text.split()),
                "coherence_score": self._calculate_coherence(chunk_text),
                "trust_score": self._calculate_textbook_trust(doc_id, subject),
            }

            if TEXT_PROCESSING_AVAILABLE:
                chunk_data["embedding"] = self.embeddings_model.encode(chunk_text)

            chunks.append(chunk_data)

        return chunks

    def _simple_chunk(self, text: str, doc_id: str) -> list[dict[str, Any]]:
        """Simple text chunking fallback."""
        chunk_size = 2000
        overlap = 200

        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            chunk_text = " ".join(chunk_words)

            chunk_data = {
                "chunk_id": f"{doc_id}_chunk_{len(chunks)}",
                "document_id": doc_id,
                "text": chunk_text,
                "position": len(chunks),
                "start_idx": i,
                "end_idx": min(i + chunk_size, len(words)),
                "subject": "unknown",
                "keywords": [],
                "entities": [],
                "word_count": len(chunk_words),
                "coherence_score": 0.7,
                "trust_score": 0.8,
            }

            chunks.append(chunk_data)

        return chunks

    def _extract_keywords(self, text: str, subject: str) -> list[str]:
        """Extract keywords from text chunk."""
        # Subject-specific keyword patterns
        keyword_patterns = {
            "mathematics": [
                "theorem",
                "proof",
                "equation",
                "formula",
                "derivative",
                "integral",
                "function",
            ],
            "physics": [
                "force",
                "energy",
                "momentum",
                "wave",
                "particle",
                "field",
                "quantum",
            ],
            "chemistry": [
                "molecule",
                "atom",
                "bond",
                "reaction",
                "compound",
                "element",
                "ion",
            ],
            "engineering": [
                "design",
                "system",
                "process",
                "material",
                "stress",
                "load",
                "efficiency",
            ],
            "computer_science": [
                "algorithm",
                "data",
                "function",
                "class",
                "method",
                "variable",
                "loop",
            ],
        }

        patterns = keyword_patterns.get(subject, [])
        found_keywords = []

        text_lower = text.lower()
        for pattern in patterns:
            if pattern in text_lower:
                found_keywords.append(pattern)

        # Add general academic keywords
        general_keywords = [
            "analysis",
            "method",
            "approach",
            "theory",
            "principle",
            "concept",
            "model",
        ]
        for keyword in general_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)

        return list(set(found_keywords))[:10]  # Limit to top 10

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities from text."""
        # Simple entity extraction - could be enhanced with NLP libraries
        entities = []

        # Numbers and formulas
        import re

        # Mathematical expressions
        math_patterns = [
            r"[a-zA-Z]\s*=\s*[^\\s]+",  # Variables like x = 5
            r"\\b\\d+\\.\\d+\\b",  # Decimal numbers
            r"\\b[A-Z][a-z]*\\s+\\d+\\b",  # Chapter/Section references
        ]

        for pattern in math_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches[:5])  # Limit matches

        return list(set(entities))[:10]  # Limit to top 10

    def _calculate_coherence(self, text: str) -> float:
        """Calculate text coherence score."""
        # Simple coherence calculation
        sentences = text.split(".")
        if len(sentences) < 2:
            return 0.5

        # Factor in sentence length variation (more coherent if consistent)
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.5

        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)

        # Lower variance indicates better coherence
        coherence = max(0.3, 1.0 - (variance / (avg_length**2)))

        return min(coherence, 1.0)

    def _calculate_textbook_trust(self, doc_id: str, subject: str) -> float:
        """Calculate trust score for textbook."""
        base_trust = 0.8  # Textbooks generally high trust

        # Subject-based adjustments
        subject_trust = {
            "mathematics": 0.9,
            "physics": 0.85,
            "engineering": 0.85,
            "chemistry": 0.85,
            "computer_science": 0.8,
            "economics": 0.75,
        }

        subject_bonus = subject_trust.get(subject, 0.8)

        # Publisher/source indicators (if detectable in filename)
        doc_lower = doc_id.lower()
        if any(
            term in doc_lower
            for term in ["handbook", "principles", "fundamentals", "introduction"]
        ):
            base_trust += 0.1

        return min(base_trust + subject_bonus - 0.8, 1.0)

    def process_all_textbooks(self) -> None:
        """Process all extracted textbooks."""
        start_time = time.time()
        logger.info("Starting STEM textbook processing...")

        # Extract zip files
        pdf_files = self.extract_zip_files()

        if not pdf_files:
            logger.error("No PDF files extracted. Check zip files.")
            return

        logger.info(f"Processing {len(pdf_files)} textbooks...")

        for i, pdf_path in enumerate(pdf_files):
            logger.info(f"Processing {i+1}/{len(pdf_files)}: {pdf_path.name}")

            try:
                # Extract text
                extraction_result = self.extract_text_from_pdf(pdf_path)

                if "error" in extraction_result:
                    logger.warning(
                        f"Skipping {pdf_path.name}: {extraction_result['error']}"
                    )
                    continue

                text = extraction_result["text"]
                if not text.strip():
                    logger.warning(f"No text extracted from {pdf_path.name}")
                    continue

                # Create document ID
                doc_id = pdf_path.stem.replace(" ", "_").replace("-", "_")

                # Classify subject
                subject = self.classify_textbook_subject(text, pdf_path.name)

                # Store book info
                self.processed_books[doc_id] = {
                    "filename": pdf_path.name,
                    "subject": subject,
                    "pages": extraction_result["pages"],
                    "word_count": extraction_result["word_count"],
                    "character_count": extraction_result["character_count"],
                    "processing_time": time.time(),
                }

                # Chunk the textbook
                chunks = self.chunk_textbook_intelligently(text, doc_id, subject)

                # Store chunks
                for chunk in chunks:
                    self.chunks[chunk["chunk_id"]] = chunk

                # Update stats
                self.processing_stats["books_processed"] += 1
                self.processing_stats["chunks_created"] += len(chunks)
                self.processing_stats["total_pages"] += extraction_result["pages"]
                self.processing_stats["total_words"] += extraction_result["word_count"]

                logger.info(
                    f"Created {len(chunks)} chunks from {pdf_path.name} ({subject})"
                )

            except Exception as e:
                logger.exception(f"Failed to process {pdf_path.name}: {e}")
                continue

        self.processing_stats["processing_time"] = time.time() - start_time

        logger.info(
            f"Processing complete: {self.processing_stats['books_processed']} books, "
            f"{self.processing_stats['chunks_created']} chunks in "
            f"{self.processing_stats['processing_time']:.1f}s"
        )

    def export_for_rag_systems(self):
        """Export processed chunks for RAG systems."""
        logger.info("Exporting chunks for RAG systems...")

        export_dir = Path("data/stem_textbooks_rag_ready")
        export_dir.mkdir(exist_ok=True)

        # Export full text versions
        full_texts = {}
        for doc_id, book_info in self.processed_books.items():
            # Reconstruct full text from chunks
            doc_chunks = [
                chunk
                for chunk in self.chunks.values()
                if chunk["document_id"] == doc_id
            ]
            doc_chunks.sort(key=lambda x: x["position"])

            full_text = "\n\n".join(chunk["text"] for chunk in doc_chunks)
            full_texts[doc_id] = {
                "title": book_info["filename"],
                "subject": book_info["subject"],
                "text": full_text,
                "metadata": book_info,
            }

            # Save individual full text file
            with open(export_dir / f"{doc_id}_full.txt", "w", encoding="utf-8") as f:
                f.write(full_text)

            # Save chunked version
            chunks_text = "\n\n---\n\n".join(chunk["text"] for chunk in doc_chunks)
            with open(export_dir / f"{doc_id}_chunks.txt", "w", encoding="utf-8") as f:
                f.write(chunks_text)

        # Export chunk data as JSON
        chunk_export = {}
        for chunk_id, chunk_data in self.chunks.items():
            export_chunk = chunk_data.copy()
            # Convert embedding to list for JSON serialization
            if "embedding" in export_chunk and hasattr(
                export_chunk["embedding"], "tolist"
            ):
                export_chunk["embedding"] = export_chunk["embedding"].tolist()
            chunk_export[chunk_id] = export_chunk

        with open(export_dir / "chunks_metadata.json", "w") as f:
            json.dump(chunk_export, f, indent=2)

        # Export processing summary
        summary = {
            "processing_stats": self.processing_stats,
            "books_by_subject": {},
            "total_books": len(self.processed_books),
            "total_chunks": len(self.chunks),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Subject distribution
        for book_info in self.processed_books.values():
            subject = book_info["subject"]
            if subject not in summary["books_by_subject"]:
                summary["books_by_subject"][subject] = 0
            summary["books_by_subject"][subject] += 1

        with open(export_dir / "processing_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Export complete: {export_dir}")
        return export_dir

    async def integrate_with_rag_systems(self, export_dir: Path) -> None:
        """Integrate processed textbooks with existing RAG systems."""
        logger.info("Integrating with RAG systems...")

        # Copy to ingested papers directory for compatibility
        ingested_papers_dir = Path("data/ingested_papers")

        # Copy all textbook files
        for file_path in export_dir.glob("*_full.txt"):
            dest_path = ingested_papers_dir / file_path.name
            shutil.copy2(file_path, dest_path)

        for file_path in export_dir.glob("*_chunks.txt"):
            dest_path = ingested_papers_dir / file_path.name
            shutil.copy2(file_path, dest_path)

        # Run the existing graph RAG analysis
        logger.info("Running updated graph RAG analysis...")

        try:
            # Import and run the existing analysis
            os.system("python scripts/analyze_graph_rag_results.py")
        except Exception as e:
            logger.warning(f"Failed to run graph RAG analysis: {e}")

        logger.info("RAG system integration complete")


async def main() -> None:
    """Main processing function."""
    logger.info("Starting STEM textbook processing pipeline...")

    # Initialize processor
    processor = STEMTextbookProcessor()

    # Process all textbooks
    processor.process_all_textbooks()

    if not processor.processed_books:
        logger.error("No textbooks processed successfully")
        return

    # Export for RAG systems
    export_dir = processor.export_for_rag_systems()

    # Integrate with RAG systems
    await processor.integrate_with_rag_systems(export_dir)

    # Print summary
    print("\n" + "=" * 70)
    print("STEM TEXTBOOK PROCESSING COMPLETE")
    print("=" * 70)
    print(f"Books extracted: {processor.processing_stats['books_extracted']}")
    print(f"Books processed: {processor.processing_stats['books_processed']}")
    print(f"Total chunks: {processor.processing_stats['chunks_created']}")
    print(f"Total pages: {processor.processing_stats['total_pages']}")
    print(f"Total words: {processor.processing_stats['total_words']:,}")
    print(f"Processing time: {processor.processing_stats['processing_time']:.1f}s")

    print("\nSubject Distribution:")
    subject_counts = {}
    for book_info in processor.processed_books.values():
        subject = book_info["subject"]
        subject_counts[subject] = subject_counts.get(subject, 0) + 1

    for subject, count in sorted(subject_counts.items()):
        print(f"  {subject}: {count} books")

    print(f"\nExported to: {export_dir}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
