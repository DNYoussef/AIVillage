#!/usr/bin/env python3
"""Simple PDF Ingestion Script for RAG System.

Ingests all AI research papers with basic chunking:
- 147 papers from the ai_papers zip file
- Grossman Non-Newtonian Calculus PDF
- Grossman Meta-Calculus PDF
"""

import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Installing PyMuPDF...")
    os.system("pip install PyMuPDF")
    import fitz

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> tuple[str, dict]:
    """Extract text from PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        text_blocks = []
        metadata = {
            "total_pages": len(doc),
            "title": doc.metadata.get("title", Path(pdf_path).stem),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "pdf_path": pdf_path,
        }

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                text_blocks.append(f"[Page {page_num + 1}]\n{text}")

        full_text = "\n\n".join(text_blocks)
        doc.close()

        return full_text, metadata

    except Exception as e:
        logger.exception(f"Failed to extract text from {pdf_path}: {e}")
        raise


def simple_chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> list[str]:
    """Simple text chunking by character count with overlap."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        # Find end position
        end = start + chunk_size

        # If we're not at the end, try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings near the break point
            for i in range(end, max(start + chunk_size // 2, end - 200), -1):
                if text[i] in ".!?":
                    end = i + 1
                    break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position with overlap
        start = end - overlap if end < len(text) else len(text)

        if start >= len(text):
            break

    return chunks


def ingest_pdf_to_rag(pdf_path: str, category: str = "research") -> bool:
    """Ingest PDF into RAG system using simple chunking."""
    try:
        logger.info(f"Processing {pdf_path}")

        # Extract text
        text, metadata = extract_text_from_pdf(pdf_path)

        if len(text.strip()) < 100:
            logger.warning(f"Insufficient text in {pdf_path}")
            return False

        # Simple chunking
        chunks = simple_chunk_text(text)

        logger.info(f"Created {len(chunks)} chunks from {Path(pdf_path).name}")

        # For now, just save to a file (we can integrate with RAG system later)
        output_dir = Path("data/ingested_papers")
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = Path(pdf_path).stem
        safe_filename = "".join(c for c in filename if c.isalnum() or c in "._- ")

        # Save full text
        with open(output_dir / f"{safe_filename}_full.txt", "w", encoding="utf-8") as f:
            f.write("=== METADATA ===\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
            f.write(f"\n=== CONTENT ===\n{text}")

        # Save chunks
        with open(output_dir / f"{safe_filename}_chunks.txt", "w", encoding="utf-8") as f:
            f.write("=== METADATA ===\n")
            f.write(f"source: {pdf_path}\n")
            f.write(f"category: {category}\n")
            f.write(f"chunk_count: {len(chunks)}\n")
            f.write("\n=== CHUNKS ===\n\n")

            for i, chunk in enumerate(chunks):
                f.write(f"--- CHUNK {i + 1} ---\n{chunk}\n\n")

        logger.info(f"Successfully processed {filename}")
        return True

    except Exception as e:
        logger.exception(f"Failed to process {pdf_path}: {e}")
        return False


def find_all_pdfs():
    """Find all PDFs to process."""
    pdfs = []

    # 1. Extracted AI papers
    ai_papers_path = Path("/tmp/ai_papers")
    if ai_papers_path.exists():
        for pdf_path in ai_papers_path.rglob("*.pdf"):
            # Categorize based on path
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
            elif "Math" in path_str or "geometry" in path_str:
                category = "mathematics"
            else:
                category = "general_research"

            pdfs.append((str(pdf_path), category))

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
            pdfs.append((paper_path, category))

    return pdfs


def main() -> None:
    """Main function."""
    logger.info("Starting simple PDF ingestion")

    pdfs = find_all_pdfs()
    logger.info(f"Found {len(pdfs)} PDFs to process")

    successful = 0
    failed = 0

    for pdf_path, category in pdfs:
        try:
            if ingest_pdf_to_rag(pdf_path, category):
                successful += 1
            else:
                failed += 1
        except Exception as e:
            logger.exception(f"Error processing {pdf_path}: {e}")
            failed += 1

    logger.info("=== INGESTION COMPLETE ===")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful / (successful + failed) * 100:.1f}%")

    # List what was created
    output_dir = Path("data/ingested_papers")
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        logger.info(f"Created {len(files)} files in {output_dir}")


if __name__ == "__main__":
    main()
