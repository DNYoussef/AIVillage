#!/usr/bin/env python3
"""
Enhanced Wikipedia ingestion with BayesRAG, cross-context tagging, and graph relationships.
Supports hierarchical chunking with global/local context and Bayesian trust scoring.
"""

import asyncio
import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import numpy as np
import wikipediaapi as wikipedia
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GlobalContext:
    """Global context for entire document/article."""

    title: str
    summary: str  # Generated summary of entire document
    word_count: int
    categories: List[str]  # Wikipedia categories
    global_tags: List[str]  # High-level semantic tags
    trust_score: float  # Bayesian trust rating (0.0-1.0)
    citation_count: int
    edit_frequency: float  # Edits per day
    source_quality: float  # Quality of citations


@dataclass
class LocalContext:
    """Local context for document chunks."""

    chunk_id: str
    parent_title: str
    section_title: Optional[str]
    content: str
    local_summary: str  # Summary of this specific chunk
    start_position: int
    end_position: int
    local_tags: List[str]  # Specific semantic tags for this chunk
    temporal_context: Optional[str]  # Time period if applicable
    geographic_context: Optional[str]  # Location if applicable
    cross_references: List[str]  # Links to other chunks/articles


@dataclass
class GraphNode:
    """Node in the knowledge graph."""

    node_id: str
    chunk_id: str
    embedding: List[float]
    node_type: str  # 'article', 'section', 'concept'
    metadata: Dict[str, Any]


@dataclass
class GraphEdge:
    """Edge in the knowledge graph with Bayesian trust."""

    source_node: str
    target_node: str
    relationship_type: str  # 'semantic', 'temporal', 'causal', 'reference'
    trust_weight: float  # Bayesian trust score (0.0-1.0)
    evidence_count: int  # How many sources support this connection
    metadata: Dict[str, Any]


class BayesRAGWikipediaIngestion:
    """Enhanced Wikipedia ingestion with BayesRAG and graph relationships."""

    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        self.embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")

        # Initialize graph
        self.knowledge_graph = nx.DiGraph()

        # Database connections
        self.global_db_path = self.data_dir / "wikipedia_global_context.db"
        self.local_db_path = self.data_dir / "wikipedia_local_context.db"
        self.graph_db_path = self.data_dir / "wikipedia_graph.db"

        self._init_databases()

    def _init_databases(self):
        """Initialize SQLite databases for contexts and graph."""

        # Global context database
        with sqlite3.connect(self.global_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS global_contexts (
                    title TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    word_count INTEGER,
                    categories TEXT, -- JSON array
                    global_tags TEXT, -- JSON array
                    trust_score REAL,
                    citation_count INTEGER,
                    edit_frequency REAL,
                    source_quality REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

        # Local context database
        with sqlite3.connect(self.local_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS local_contexts (
                    chunk_id TEXT PRIMARY KEY,
                    parent_title TEXT NOT NULL,
                    section_title TEXT,
                    content TEXT NOT NULL,
                    local_summary TEXT NOT NULL,
                    start_position INTEGER,
                    end_position INTEGER,
                    local_tags TEXT, -- JSON array
                    temporal_context TEXT,
                    geographic_context TEXT,
                    cross_references TEXT, -- JSON array
                    embedding BLOB, -- Numpy array as bytes
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (parent_title) REFERENCES global_contexts (title)
                )
            """
            )

        # Graph database
        with sqlite3.connect(self.graph_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_nodes (
                    node_id TEXT PRIMARY KEY,
                    chunk_id TEXT,
                    embedding BLOB, -- Numpy array as bytes
                    node_type TEXT,
                    metadata TEXT, -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (chunk_id) REFERENCES local_contexts (chunk_id)
                )
            """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_edges (
                    edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_node TEXT,
                    target_node TEXT,
                    relationship_type TEXT,
                    trust_weight REAL,
                    evidence_count INTEGER,
                    metadata TEXT, -- JSON
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_node) REFERENCES graph_nodes (node_id),
                    FOREIGN KEY (target_node) REFERENCES graph_nodes (node_id)
                )
            """
            )

    def calculate_article_trust_score(self, page_data: Dict[str, Any]) -> float:
        """Calculate Bayesian trust score for Wikipedia article."""

        # Base factors for trust calculation
        factors = {
            "citation_density": 0.0,  # Citations per 1000 words
            "edit_stability": 0.0,  # Lower edit frequency = more stable
            "category_breadth": 0.0,  # More categories = more comprehensive
            "content_quality": 0.0,  # Length and structure quality
            "reference_quality": 0.0,  # Quality of external references
        }

        content = page_data.get("content", "")
        word_count = len(content.split())

        # Citation density (looking for citation patterns)
        citation_patterns = content.count("[") + content.count("{{cite")
        factors["citation_density"] = min(
            citation_patterns / max(word_count / 1000, 1), 1.0
        )

        # Content quality based on length and structure
        if word_count > 5000:
            factors["content_quality"] = 0.9
        elif word_count > 2000:
            factors["content_quality"] = 0.7
        elif word_count > 500:
            factors["content_quality"] = 0.5
        else:
            factors["content_quality"] = 0.3

        # Category breadth
        categories = page_data.get("categories", [])
        factors["category_breadth"] = min(len(categories) / 20.0, 1.0)

        # Edit stability (assume newer articles are less stable)
        factors["edit_stability"] = 0.8  # Default assumption

        # Reference quality (basic heuristic)
        factors["reference_quality"] = min(citation_patterns / 10.0, 1.0)

        # Weighted combination
        weights = {
            "citation_density": 0.3,
            "edit_stability": 0.2,
            "category_breadth": 0.15,
            "content_quality": 0.25,
            "reference_quality": 0.1,
        }

        trust_score = sum(factors[k] * weights[k] for k in factors)
        return max(0.1, min(1.0, trust_score))  # Clamp between 0.1 and 1.0

    def generate_global_summary(self, content: str, title: str) -> str:
        """Generate global summary for entire article."""
        # Simple extractive summarization (in production, use transformer model)
        sentences = content.split(". ")

        # Take first paragraph and key sentences
        first_para_end = content.find("\n\n")
        if first_para_end > 0:
            first_paragraph = content[:first_para_end]
        else:
            first_paragraph = sentences[0] if sentences else ""

        # Add title context
        summary = f"{title}: {first_paragraph}"

        # Limit to reasonable length
        if len(summary) > 300:
            summary = summary[:297] + "..."

        return summary

    def extract_global_tags(self, content: str, categories: List[str]) -> List[str]:
        """Extract high-level semantic tags from article."""
        tags = set()

        # Add category-based tags
        for category in categories:
            if any(term in category.lower() for term in ["country", "nation"]):
                tags.add("geography")
            elif any(term in category.lower() for term in ["history", "historical"]):
                tags.add("history")
            elif any(
                term in category.lower() for term in ["science", "physics", "biology"]
            ):
                tags.add("science")
            elif any(
                term in category.lower() for term in ["art", "culture", "literature"]
            ):
                tags.add("culture")
            elif any(term in category.lower() for term in ["politics", "government"]):
                tags.add("politics")

        # Content-based tag extraction (keyword matching)
        content_lower = content.lower()

        # Temporal indicators
        if any(year in content for year in ["1800", "1900", "19th", "20th"]):
            tags.add("historical-period")

        # Geographic indicators
        if any(
            geo in content_lower for geo in ["country", "city", "region", "continent"]
        ):
            tags.add("geography")

        # Scientific indicators
        if any(
            sci in content_lower
            for sci in ["research", "study", "theory", "experiment"]
        ):
            tags.add("science")

        return list(tags)

    def create_hierarchical_chunks(
        self, content: str, title: str, chunk_size: int = 1000
    ) -> List[LocalContext]:
        """Create hierarchical chunks with local context."""
        chunks = []

        # Split by sections first (looking for == Section == patterns)
        sections = self._split_by_sections(content)

        chunk_id_counter = 0

        for section_title, section_content in sections:
            # Further split long sections into smaller chunks
            section_chunks = self._split_into_chunks(section_content, chunk_size)

            for i, chunk_content in enumerate(section_chunks):
                chunk_id = f"{title}_{chunk_id_counter:04d}"
                chunk_id_counter += 1

                # Generate local summary
                local_summary = self._generate_local_summary(
                    chunk_content, section_title
                )

                # Extract local tags
                local_tags = self._extract_local_tags(chunk_content, section_title)

                # Extract temporal and geographic context
                temporal_context = self._extract_temporal_context(chunk_content)
                geographic_context = self._extract_geographic_context(chunk_content)

                # Find cross-references
                cross_references = self._extract_cross_references(chunk_content)

                local_context = LocalContext(
                    chunk_id=chunk_id,
                    parent_title=title,
                    section_title=section_title,
                    content=chunk_content,
                    local_summary=local_summary,
                    start_position=0,  # Would need to calculate actual position
                    end_position=len(chunk_content),
                    local_tags=local_tags,
                    temporal_context=temporal_context,
                    geographic_context=geographic_context,
                    cross_references=cross_references,
                )

                chunks.append(local_context)

        return chunks

    def _split_by_sections(self, content: str) -> List[Tuple[str, str]]:
        """Split content by Wikipedia sections."""
        sections = []

        # Look for == Section == patterns
        lines = content.split("\n")
        current_section = "Introduction"
        current_content = []

        for line in lines:
            if line.startswith("==") and line.endswith("=="):
                # Save previous section
                if current_content:
                    sections.append((current_section, "\n".join(current_content)))

                # Start new section
                current_section = line.strip("= ")
                current_content = []
            else:
                current_content.append(line)

        # Add final section
        if current_content:
            sections.append((current_section, "\n".join(current_content)))

        return sections

    def _split_into_chunks(self, content: str, chunk_size: int) -> List[str]:
        """Split content into overlapping chunks."""
        words = content.split()
        chunks = []
        overlap = chunk_size // 4  # 25% overlap

        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i : i + chunk_size]
            chunks.append(" ".join(chunk_words))

            if i + chunk_size >= len(words):
                break

        return chunks

    def _generate_local_summary(self, content: str, section_title: str) -> str:
        """Generate local summary for chunk."""
        # Simple extractive approach - take first sentence and key information
        sentences = content.split(". ")

        if sentences:
            first_sentence = sentences[0]
            if section_title and section_title != "Introduction":
                summary = f"{section_title}: {first_sentence}"
            else:
                summary = first_sentence

            if len(summary) > 200:
                summary = summary[:197] + "..."

            return summary

        return f"Section about {section_title}" if section_title else "Content chunk"

    def _extract_local_tags(self, content: str, section_title: str) -> List[str]:
        """Extract specific semantic tags for chunk."""
        tags = set()
        content_lower = content.lower()

        # Section-based tags
        if section_title:
            section_lower = section_title.lower()
            if any(term in section_lower for term in ["history", "historical"]):
                tags.add("history")
            elif any(term in section_lower for term in ["geography", "location"]):
                tags.add("geography")
            elif any(term in section_lower for term in ["culture", "society"]):
                tags.add("culture")
            elif any(term in section_lower for term in ["politics", "government"]):
                tags.add("politics")
            elif any(term in section_lower for term in ["economy", "economic"]):
                tags.add("economics")

        # Content-based specific tags
        if any(term in content_lower for term in ["war", "battle", "conflict"]):
            tags.add("military-conflict")
        elif any(term in content_lower for term in ["king", "emperor", "ruler"]):
            tags.add("leadership")
        elif any(term in content_lower for term in ["trade", "commerce", "industry"]):
            tags.add("economics")
        elif any(term in content_lower for term in ["art", "painting", "sculpture"]):
            tags.add("visual-arts")

        return list(tags)

    def _extract_temporal_context(self, content: str) -> Optional[str]:
        """Extract time period context from chunk."""
        import re

        # Look for year patterns
        year_patterns = re.findall(r"\b(1[0-9]{3}|20[0-2][0-9])\b", content)

        if year_patterns:
            years = [int(y) for y in year_patterns]
            min_year = min(years)
            max_year = max(years)

            if min_year == max_year:
                return str(min_year)
            else:
                return f"{min_year}-{max_year}"

        # Look for period indicators
        if any(
            period in content.lower()
            for period in ["medieval", "renaissance", "industrial revolution"]
        ):
            for period in ["medieval", "renaissance", "industrial revolution"]:
                if period in content.lower():
                    return period

        return None

    def _extract_geographic_context(self, content: str) -> Optional[str]:
        """Extract geographic context from chunk."""
        import re

        # Look for country/city patterns (simplified)
        geo_patterns = [
            "Germany",
            "France",
            "England",
            "Italy",
            "Spain",
            "Europe",
            "Asia",
            "Africa",
        ]

        for geo in geo_patterns:
            if geo in content:
                return geo

        return None

    def _extract_cross_references(self, content: str) -> List[str]:
        """Extract references to other articles/concepts."""
        import re

        # Look for Wikipedia link patterns [[Article Name]]
        link_pattern = r"\[\[(.*?)\]\]"
        links = re.findall(link_pattern, content)

        # Clean and deduplicate
        cross_refs = []
        for link in links[:10]:  # Limit to first 10 links
            # Remove disambiguation
            if "|" in link:
                link = link.split("|")[0]
            cross_refs.append(link.strip())

        return list(set(cross_refs))

    async def ingest_article_with_bayesrag(
        self, title: str
    ) -> Tuple[GlobalContext, List[LocalContext]]:
        """Ingest single Wikipedia article with full BayesRAG processing."""
        logger.info(f"Ingesting article: {title}")

        try:
            # Initialize Wikipedia API
            wiki_wiki = wikipedia.Wikipedia(
                language="en", user_agent="BayesRAG/1.0 (AIVillage Research Project)"
            )

            # Fetch Wikipedia article
            page = wiki_wiki.page(title)

            if not page.exists():
                raise ValueError(f"Wikipedia page '{title}' does not exist")

            # Create article data structure
            page_data = {
                "title": title,
                "content": page.text,
                "categories": list(page.categories.keys()) if page.categories else [],
                "summary": page.summary,
            }

            # Calculate trust score
            trust_score = self.calculate_article_trust_score(page_data)

            # Generate global context
            global_summary = self.generate_global_summary(page_data["content"], title)
            global_tags = self.extract_global_tags(
                page_data["content"], page_data["categories"]
            )

            global_context = GlobalContext(
                title=title,
                summary=global_summary,
                word_count=len(page_data["content"].split()),
                categories=page_data["categories"],
                global_tags=global_tags,
                trust_score=trust_score,
                citation_count=page_data["content"].count("["),  # Rough approximation
                edit_frequency=0.1,  # Would need API call to get real data
                source_quality=trust_score,  # Correlated for now
            )

            # Create hierarchical chunks
            local_contexts = self.create_hierarchical_chunks(
                page_data["content"], title
            )

            # Generate embeddings for chunks
            for local_context in local_contexts:
                embedding = self.embedder.encode(local_context.content)
                # Store embedding separately since dataclass doesn't handle numpy arrays well
                setattr(local_context, "_embedding", embedding)

            # Store in databases
            await self._store_contexts(global_context, local_contexts)

            logger.info(
                f"Successfully ingested {title}: {len(local_contexts)} chunks, trust={trust_score:.3f}"
            )

            return global_context, local_contexts

        except Exception as e:
            logger.error(f"Error ingesting {title}: {e}")
            raise

    async def _store_contexts(
        self, global_context: GlobalContext, local_contexts: List[LocalContext]
    ):
        """Store global and local contexts in databases."""

        # Store global context
        with sqlite3.connect(self.global_db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO global_contexts
                (title, summary, word_count, categories, global_tags, trust_score,
                 citation_count, edit_frequency, source_quality, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    global_context.title,
                    global_context.summary,
                    global_context.word_count,
                    json.dumps(global_context.categories),
                    json.dumps(global_context.global_tags),
                    global_context.trust_score,
                    global_context.citation_count,
                    global_context.edit_frequency,
                    global_context.source_quality,
                    datetime.now().isoformat(),
                ),
            )

        # Store local contexts
        with sqlite3.connect(self.local_db_path) as conn:
            for local_context in local_contexts:
                embedding_bytes = getattr(
                    local_context, "_embedding", np.array([])
                ).tobytes()

                conn.execute(
                    """
                    INSERT OR REPLACE INTO local_contexts
                    (chunk_id, parent_title, section_title, content, local_summary,
                     start_position, end_position, local_tags, temporal_context,
                     geographic_context, cross_references, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        local_context.chunk_id,
                        local_context.parent_title,
                        local_context.section_title,
                        local_context.content,
                        local_context.local_summary,
                        local_context.start_position,
                        local_context.end_position,
                        json.dumps(local_context.local_tags),
                        local_context.temporal_context,
                        local_context.geographic_context,
                        json.dumps(local_context.cross_references),
                        embedding_bytes,
                    ),
                )


async def main():
    """Test the BayesRAG Wikipedia ingestion system."""

    ingestion = BayesRAGWikipediaIngestion()

    # Test with a few articles
    test_articles = [
        "Germany",
        "World War I",
        "Industrial Revolution",
        "Leonardo da Vinci",
    ]

    logger.info("Starting BayesRAG Wikipedia ingestion test")

    for article in test_articles:
        try:
            start_time = time.time()
            global_ctx, local_ctxs = await ingestion.ingest_article_with_bayesrag(
                article
            )

            processing_time = time.time() - start_time

            print(f"\n=== {article} ===")
            print(f"Trust Score: {global_ctx.trust_score:.3f}")
            print(f"Global Tags: {global_ctx.global_tags}")
            print(f"Chunks Created: {len(local_ctxs)}")
            print(f"Processing Time: {processing_time:.2f}s")
            print(f"Global Summary: {global_ctx.summary[:100]}...")

            # Show sample local context
            if local_ctxs:
                sample_chunk = local_ctxs[0]
                print(f"Sample Chunk: {sample_chunk.local_summary}")
                print(f"Local Tags: {sample_chunk.local_tags}")
                print(f"Temporal: {sample_chunk.temporal_context}")
                print(f"Geographic: {sample_chunk.geographic_context}")

        except Exception as e:
            logger.error(f"Failed to ingest {article}: {e}")

    logger.info("BayesRAG Wikipedia ingestion test completed")


if __name__ == "__main__":
    asyncio.run(main())
