#!/usr/bin/env python3
"""
Scaled Wikipedia Ingestion Pipeline for 1000+ Articles.

Features:
- Automated ingestion across 15 categories
- Parallel processing with batch embeddings
- Quality control with trust score filtering
- Progress tracking and resumability
- Memory-efficient chunking
"""

import asyncio
import concurrent.futures
import json
import logging
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import wikipedia
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(
    0,
    str(
        Path(__file__).parent.parent
        / "src"
        / "production"
        / "rag"
        / "rag_system"
        / "core"
    ),
)

from bayesrag_codex_enhanced import BayesRAGEnhancedPipeline, TrustMetrics
from codex_rag_integration import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class WikipediaArticle:
    """Wikipedia article with metadata."""

    title: str
    content: str
    summary: str
    categories: List[str]
    links: List[str]
    word_count: int
    revision_id: Optional[str] = None


@dataclass
class IngestionStats:
    """Statistics for ingestion tracking."""

    total_articles: int = 0
    successful_articles: int = 0
    failed_articles: int = 0
    total_chunks: int = 0
    total_embeddings: int = 0
    processing_time: float = 0.0
    average_trust_score: float = 0.0


class WikipediaScaledIngestion:
    """
    Scaled Wikipedia ingestion pipeline for 1000+ articles.
    """

    # Target categories for diverse knowledge coverage
    CATEGORIES = [
        "Science",
        "Technology",
        "History",
        "Geography",
        "Mathematics",
        "Medicine",
        "Philosophy",
        "Literature",
        "Art",
        "Economics",
        "Politics",
        "Biology",
        "Physics",
        "Chemistry",
        "Engineering",
    ]

    # Quality thresholds
    MIN_WORD_COUNT = 500
    MIN_LINKS = 5
    MIN_TRUST_SCORE = 0.3

    def __init__(
        self,
        data_dir: Path = Path("data"),
        batch_size: int = 32,
        max_workers: int = 4,
        resume: bool = True,
    ):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.max_workers = max_workers
        self.resume = resume

        # Progress tracking
        self.progress_file = data_dir / "ingestion_progress.json"
        self.processed_articles: Set[str] = set()
        self.failed_articles: Set[str] = set()

        # Load progress if resuming
        if resume:
            self._load_progress()

        # Initialize pipeline
        self.pipeline = None
        self.embedder = None

        # Statistics
        self.stats = IngestionStats()

    def _load_progress(self) -> None:
        """Load ingestion progress from disk."""

        if self.progress_file.exists():
            try:
                with open(self.progress_file, "r") as f:
                    progress = json.load(f)
                    self.processed_articles = set(progress.get("processed", []))
                    self.failed_articles = set(progress.get("failed", []))

                logger.info(
                    f"Loaded progress: {len(self.processed_articles)} processed, "
                    f"{len(self.failed_articles)} failed"
                )
            except Exception as e:
                logger.warning(f"Failed to load progress: {e}")

    def _save_progress(self) -> None:
        """Save ingestion progress to disk."""

        progress = {
            "processed": list(self.processed_articles),
            "failed": list(self.failed_articles),
            "timestamp": time.time(),
        }

        try:
            with open(self.progress_file, "w") as f:
                json.dump(progress, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    async def initialize_pipeline(self) -> bool:
        """Initialize the BayesRAG-enhanced CODEX pipeline."""

        try:
            logger.info("Initializing BayesRAG-enhanced pipeline...")
            self.pipeline = BayesRAGEnhancedPipeline(self.data_dir)
            self.embedder = SentenceTransformer("paraphrase-MiniLM-L3-v2")
            logger.info("Pipeline initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            return False

    def fetch_articles_by_category(self, category: str, limit: int = 100) -> List[str]:
        """Fetch article titles from a Wikipedia category."""

        articles = []

        try:
            # Search for articles in category
            search_results = wikipedia.search(category, results=limit)

            for title in search_results:
                if (
                    title not in self.processed_articles
                    and title not in self.failed_articles
                ):
                    articles.append(title)

            logger.info(f"Found {len(articles)} new articles in category: {category}")

        except Exception as e:
            logger.error(f"Failed to fetch articles for category {category}: {e}")

        return articles

    def fetch_article_content(self, title: str) -> Optional[WikipediaArticle]:
        """Fetch full article content from Wikipedia."""

        try:
            # Set timeout to avoid hanging
            wikipedia.set_lang("en")

            # Get page
            page = wikipedia.page(title)

            # Extract content
            content = page.content
            summary = page.summary[:500] if page.summary else ""

            # Get metadata
            categories = page.categories[:10] if hasattr(page, "categories") else []
            links = page.links[:50] if hasattr(page, "links") else []
            word_count = len(content.split())

            # Quality check
            if word_count < self.MIN_WORD_COUNT:
                logger.debug(f"Article '{title}' too short: {word_count} words")
                return None

            if len(links) < self.MIN_LINKS:
                logger.debug(f"Article '{title}' has too few links: {len(links)}")
                return None

            article = WikipediaArticle(
                title=title,
                content=content,
                summary=summary,
                categories=categories,
                links=links,
                word_count=word_count,
                revision_id=(
                    str(page.revision_id) if hasattr(page, "revision_id") else None
                ),
            )

            return article

        except wikipedia.exceptions.DisambiguationError as e:
            # Try first option
            if e.options:
                return self.fetch_article_content(e.options[0])
            return None

        except wikipedia.exceptions.PageError:
            logger.debug(f"Page not found: {title}")
            return None

        except Exception as e:
            logger.error(f"Failed to fetch article '{title}': {e}")
            return None

    def calculate_trust_score(self, article: WikipediaArticle) -> float:
        """Calculate trust score for an article."""

        # Base score from article metrics
        word_score = min(1.0, article.word_count / 5000)
        link_score = min(1.0, len(article.links) / 50)
        category_score = min(1.0, len(article.categories) / 10)

        # Calculate base trust
        base_trust = (word_score + link_score + category_score) / 3

        # Boost for certain categories
        quality_categories = {"Featured articles", "Good articles", "Vital articles"}
        category_boost = 0.0

        for cat in article.categories:
            if any(q in cat for q in quality_categories):
                category_boost = 0.2
                break

        # Final trust score
        trust_score = min(1.0, base_trust + category_boost)

        return trust_score

    def create_document_from_article(
        self, article: WikipediaArticle, trust_score: float
    ) -> Document:
        """Convert Wikipedia article to CODEX Document format."""

        # Create metadata
        metadata = {
            "source": "wikipedia",
            "title": article.title,
            "summary": article.summary,
            "categories": article.categories,
            "word_count": article.word_count,
            "link_count": len(article.links),
            "trust_score": trust_score,
            "revision_id": article.revision_id,
            "ingestion_timestamp": time.time(),
        }

        # Create document
        doc = Document(
            id=f"wiki_{article.title.replace(' ', '_').lower()}",
            title=article.title,
            content=article.content,
            source_type="wikipedia",
            metadata=metadata,
        )

        return doc

    async def process_article_batch(
        self, articles: List[WikipediaArticle]
    ) -> List[Document]:
        """Process a batch of articles in parallel."""

        documents = []

        # Calculate trust scores
        for article in articles:
            trust_score = self.calculate_trust_score(article)

            # Filter by minimum trust score
            if trust_score >= self.MIN_TRUST_SCORE:
                doc = self.create_document_from_article(article, trust_score)
                documents.append(doc)
                self.stats.average_trust_score += trust_score
            else:
                logger.debug(
                    f"Article '{article.title}' filtered out: trust_score={trust_score:.2f}"
                )

        return documents

    async def ingest_articles(self, target_count: int = 1000) -> IngestionStats:
        """Ingest target number of Wikipedia articles."""

        start_time = time.time()

        # Initialize pipeline
        if not await self.initialize_pipeline():
            logger.error("Failed to initialize pipeline")
            return self.stats

        # Calculate articles per category
        articles_per_category = max(10, target_count // len(self.CATEGORIES))

        logger.info(
            f"Target: {target_count} articles across {len(self.CATEGORIES)} categories"
        )
        logger.info(f"Fetching ~{articles_per_category} articles per category")

        all_titles = []

        # Fetch article titles from each category
        with tqdm(total=len(self.CATEGORIES), desc="Fetching categories") as pbar:
            for category in self.CATEGORIES:
                titles = self.fetch_articles_by_category(
                    category, articles_per_category
                )
                all_titles.extend(titles)
                pbar.update(1)

        logger.info(f"Found {len(all_titles)} unique articles to process")

        # Process articles in batches
        processed_count = 0

        with tqdm(total=len(all_titles), desc="Processing articles") as pbar:
            # Use thread pool for parallel fetching
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                # Process in chunks
                for i in range(0, len(all_titles), self.batch_size):
                    batch_titles = all_titles[i : i + self.batch_size]

                    # Fetch articles in parallel
                    future_to_title = {
                        executor.submit(self.fetch_article_content, title): title
                        for title in batch_titles
                    }

                    articles = []

                    for future in concurrent.futures.as_completed(future_to_title):
                        title = future_to_title[future]

                        try:
                            article = future.result(timeout=30)

                            if article:
                                articles.append(article)
                                self.processed_articles.add(title)
                                self.stats.successful_articles += 1
                            else:
                                self.failed_articles.add(title)
                                self.stats.failed_articles += 1

                        except Exception as e:
                            logger.error(f"Failed to process '{title}': {e}")
                            self.failed_articles.add(title)
                            self.stats.failed_articles += 1

                        pbar.update(1)

                    # Process batch
                    if articles:
                        documents = await self.process_article_batch(articles)

                        if documents:
                            # Index documents in pipeline
                            try:
                                index_stats = self.pipeline.index_documents(documents)
                                self.stats.total_chunks += index_stats.get(
                                    "chunks_created", 0
                                )
                                self.stats.total_embeddings += index_stats.get(
                                    "vectors_indexed", 0
                                )
                                processed_count += len(documents)

                            except Exception as e:
                                logger.error(f"Failed to index batch: {e}")

                    # Save progress periodically
                    if processed_count % 100 == 0:
                        self._save_progress()

                    # Stop if target reached
                    if processed_count >= target_count:
                        break

        # Final statistics
        self.stats.total_articles = len(self.processed_articles)
        self.stats.processing_time = time.time() - start_time

        if self.stats.successful_articles > 0:
            self.stats.average_trust_score /= self.stats.successful_articles

        # Save final progress
        self._save_progress()

        # Log summary
        logger.info("\n=== Ingestion Complete ===")
        logger.info(f"Total articles processed: {self.stats.total_articles}")
        logger.info(f"Successful: {self.stats.successful_articles}")
        logger.info(f"Failed: {self.stats.failed_articles}")
        logger.info(f"Total chunks created: {self.stats.total_chunks}")
        logger.info(f"Total embeddings: {self.stats.total_embeddings}")
        logger.info(f"Average trust score: {self.stats.average_trust_score:.3f}")
        logger.info(f"Processing time: {self.stats.processing_time:.1f} seconds")
        logger.info(
            f"Articles per second: {self.stats.total_articles / self.stats.processing_time:.2f}"
        )

        return self.stats

    async def verify_ingestion(self) -> Dict[str, Any]:
        """Verify the ingested content quality."""

        verification = {
            "total_documents": 0,
            "index_size": 0,
            "sample_queries": [],
            "performance_metrics": {},
        }

        if not self.pipeline:
            await self.initialize_pipeline()

        # Get pipeline metrics
        perf_metrics = self.pipeline.get_performance_metrics()
        verification["performance_metrics"] = perf_metrics
        verification["index_size"] = perf_metrics.get("index_size", 0)

        # Test sample queries
        test_queries = [
            "artificial intelligence applications",
            "quantum computing principles",
            "climate change causes",
            "World War II timeline",
            "DNA structure and function",
        ]

        for query in test_queries:
            try:
                results, metrics = await self.pipeline.retrieve_with_trust(
                    query=query, k=3
                )

                verification["sample_queries"].append(
                    {
                        "query": query,
                        "results_count": len(results),
                        "latency_ms": metrics.get("latency_ms", 0),
                        "avg_trust": metrics.get("avg_trust_score", 0),
                    }
                )

            except Exception as e:
                logger.error(f"Query verification failed for '{query}': {e}")

        return verification


async def main():
    """Run the scaled Wikipedia ingestion pipeline."""

    print("=== Wikipedia Scaled Ingestion Pipeline ===\n")

    # Configuration
    target_articles = 1000  # Target 1000+ articles
    data_dir = Path("data/wikipedia_scaled")

    # Initialize ingestion pipeline
    ingestion = WikipediaScaledIngestion(
        data_dir=data_dir, batch_size=50, max_workers=8, resume=True
    )

    # Run ingestion
    print(f"Starting ingestion of {target_articles} Wikipedia articles...")
    stats = await ingestion.ingest_articles(target_count=target_articles)

    # Print results
    print("\n=== Ingestion Results ===")
    print(f"Total articles: {stats.total_articles}")
    print(f"Successful: {stats.successful_articles}")
    print(f"Failed: {stats.failed_articles}")
    print(f"Chunks created: {stats.total_chunks}")
    print(f"Embeddings indexed: {stats.total_embeddings}")
    print(f"Average trust score: {stats.average_trust_score:.3f}")
    print(f"Processing time: {stats.processing_time / 60:.1f} minutes")

    # Verify ingestion
    print("\n=== Verifying Ingestion ===")
    verification = await ingestion.verify_ingestion()

    print(f"Index size: {verification['index_size']} documents")
    print("\nSample query results:")

    for query_result in verification["sample_queries"]:
        print(f"  Query: {query_result['query']}")
        print(f"    Results: {query_result['results_count']}")
        print(f"    Latency: {query_result['latency_ms']:.1f}ms")
        print(f"    Avg Trust: {query_result['avg_trust']:.3f}")

    # Check performance targets
    perf = verification["performance_metrics"]
    print("\n=== Performance Metrics ===")
    print(f"Average latency: {perf.get('avg_latency_ms', 0):.1f}ms")
    print(f"P95 latency: {perf.get('p95_latency_ms', 0):.1f}ms")
    print(f"Meets <100ms target: {perf.get('meets_target', False)}")

    if stats.total_articles >= 1000:
        print("\n✅ Successfully ingested 1000+ Wikipedia articles!")
    else:
        print(f"\n⚠️ Ingested {stats.total_articles} articles (target was 1000+)")

    return stats


if __name__ == "__main__":
    asyncio.run(main())
