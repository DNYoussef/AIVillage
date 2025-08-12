#!/usr/bin/env python3
"""Wikipedia Article Ingestion Script for RAG Pipeline."""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import requests

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
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

from codex_rag_integration import CODEXRAGPipeline, Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WikipediaIngestion:
    """Wikipedia article ingestion for RAG pipeline."""

    def __init__(self, target_articles: int = 1000) -> None:
        self.target_articles = target_articles
        self.rag_pipeline = None

        # Wikipedia API base URLs
        self.wikipedia_api = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        self.wikipedia_search = "https://en.wikipedia.org/w/api.php"

        # Educational/STEM categories for quality content
        self.categories = [
            "Science",
            "Technology",
            "Mathematics",
            "Physics",
            "Chemistry",
            "Biology",
            "Computer_science",
            "Engineering",
            "Medicine",
            "Education",
            "History",
            "Geography",
            "Literature",
            "Philosophy",
            "Psychology",
            "Economics",
            "Artificial_intelligence",
            "Machine_learning",
            "Space",
            "Climate_change",
            "Renewable_energy",
            "Quantum_physics",
            "Genetics",
        ]

    async def initialize_rag_pipeline(self) -> bool:
        """Initialize the CODEX RAG pipeline."""
        try:
            self.rag_pipeline = CODEXRAGPipeline()
            logger.info("RAG pipeline initialized successfully")
            return True
        except Exception as e:
            logger.exception(f"Failed to initialize RAG pipeline: {e}")
            return False

    def search_articles_by_category(self, category: str, limit: int = 50) -> list[str]:
        """Search for articles in a specific category."""
        try:
            params = {
                "action": "query",
                "format": "json",
                "list": "categorymembers",
                "cmtitle": f"Category:{category}",
                "cmlimit": limit,
                "cmnamespace": "0",  # Main namespace (articles)
            }

            response = requests.get(self.wikipedia_search, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            titles = []
            if "query" in data and "categorymembers" in data["query"]:
                titles = [page["title"] for page in data["query"]["categorymembers"]]

            logger.info(f"Found {len(titles)} articles in category '{category}'")
            return titles

        except Exception as e:
            logger.exception(f"Failed to search category '{category}': {e}")
            return []

    def fetch_article_content(self, title: str) -> dict[str, Any]:
        """Fetch full article content from Wikipedia."""
        try:
            # First get the summary
            summary_url = self.wikipedia_api + title.replace(" ", "_")
            summary_response = requests.get(summary_url, timeout=10)
            summary_response.raise_for_status()
            summary_data = summary_response.json()

            # Get full content using the extract API
            params = {
                "action": "query",
                "format": "json",
                "titles": title,
                "prop": "extracts",
                "exintro": False,
                "explaintext": True,
                "exsectionformat": "plain",
            }

            content_response = requests.get(
                self.wikipedia_search, params=params, timeout=10
            )
            content_response.raise_for_status()
            content_data = content_response.json()

            # Extract content
            pages = content_data.get("query", {}).get("pages", {})
            page_id = next(iter(pages.keys()))
            full_content = pages[page_id].get("extract", "")

            # Fallback to summary if no full content
            if not full_content.strip():
                full_content = summary_data.get("extract", "")

            return {
                "title": title,
                "content": full_content,
                "summary": summary_data.get("extract", ""),
                "url": summary_data.get("content_urls", {})
                .get("desktop", {})
                .get("page", ""),
                "word_count": len(full_content.split()),
                "extract_date": time.strftime("%Y-%m-%d"),
            }

        except Exception as e:
            logger.warning(f"Failed to fetch article '{title}': {e}")
            return None

    def filter_quality_articles(
        self, articles: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Filter articles for quality content suitable for education."""
        quality_articles = []

        for article in articles:
            if not article:
                continue

            # Quality filters
            word_count = article.get("word_count", 0)
            content = article.get("content", "")
            title = article.get("title", "")

            # Skip articles that are too short or too long
            if word_count < 100 or word_count > 10000:
                continue

            # Skip disambiguation pages and lists
            if any(
                skip_term in title.lower()
                for skip_term in [
                    "disambiguation",
                    "list of",
                    "category:",
                    "template:",
                    "user:",
                ]
            ):
                continue

            # Skip articles with too little content
            if len(content.strip()) < 500:
                continue

            quality_articles.append(article)

        return quality_articles

    async def ingest_articles(self, articles: list[dict[str, Any]]) -> dict[str, Any]:
        """Ingest articles into the RAG pipeline."""
        if not self.rag_pipeline:
            msg = "RAG pipeline not initialized"
            raise RuntimeError(msg)

        documents = []
        for article in articles:
            if not article:
                continue

            doc = Document(
                id=f"wiki_{article['title'].replace(' ', '_')}",
                title=article["title"],
                content=article["content"],
                source_type="wikipedia",
                metadata={
                    "url": article.get("url", ""),
                    "word_count": article.get("word_count", 0),
                    "extract_date": article.get("extract_date", ""),
                    "summary": article.get("summary", "")[:500],  # Truncated summary
                },
            )
            documents.append(doc)

        # Index documents in batches
        batch_size = 50
        total_stats = {
            "documents_processed": 0,
            "chunks_created": 0,
            "vectors_indexed": 0,
            "processing_time_ms": 0,
        }

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            logger.info(
                f"Indexing batch {i // batch_size + 1} ({len(batch)} documents)"
            )

            batch_stats = self.rag_pipeline.index_documents(batch)

            # Accumulate stats
            for key in total_stats:
                total_stats[key] += batch_stats.get(key, 0)

            # Small delay between batches to avoid overwhelming the system
            await asyncio.sleep(1)

        return total_stats

    async def run_ingestion(self) -> dict[str, Any]:
        """Run the complete Wikipedia ingestion process."""
        logger.info(
            f"Starting Wikipedia ingestion (target: {self.target_articles} articles)"
        )

        # Initialize RAG pipeline
        if not await self.initialize_rag_pipeline():
            msg = "Failed to initialize RAG pipeline"
            raise RuntimeError(msg)

        all_articles = []
        articles_per_category = self.target_articles // len(self.categories) + 10

        # Search articles across categories
        for category in self.categories:
            if len(all_articles) >= self.target_articles:
                break

            logger.info(f"Searching category: {category}")
            titles = self.search_articles_by_category(category, articles_per_category)

            # Fetch article content
            category_articles = []
            for title in titles[:articles_per_category]:
                if len(all_articles) + len(category_articles) >= self.target_articles:
                    break

                article = self.fetch_article_content(title)
                if article:
                    category_articles.append(article)

                # Rate limiting
                time.sleep(0.1)

            # Filter for quality
            quality_articles = self.filter_quality_articles(category_articles)
            all_articles.extend(quality_articles)

            logger.info(
                f"Category '{category}': {len(quality_articles)} quality articles"
            )

        # Trim to target number
        all_articles = all_articles[: self.target_articles]
        logger.info(f"Collected {len(all_articles)} articles for ingestion")

        # Ingest into RAG pipeline
        stats = await self.ingest_articles(all_articles)

        # Get final performance metrics
        performance_metrics = self.rag_pipeline.get_performance_metrics()

        final_stats = {
            **stats,
            "articles_ingested": len(all_articles),
            "average_article_length": sum(a.get("word_count", 0) for a in all_articles)
            // len(all_articles),
            "performance_metrics": performance_metrics,
        }

        logger.info(f"Ingestion complete: {final_stats}")
        return final_stats


async def main():
    """Main function to run Wikipedia ingestion."""
    try:
        # Get target from command line args or use default
        target = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

        ingestion = WikipediaIngestion(target_articles=target)
        stats = await ingestion.run_ingestion()

        # Save results
        results_file = Path("data/wikipedia_ingestion_results.json")
        results_file.parent.mkdir(parents=True, exist_ok=True)

        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, default=str)

        print("\n=== Wikipedia Ingestion Complete ===")
        print(f"Articles ingested: {stats['articles_ingested']}")
        print(f"Documents processed: {stats['documents_processed']}")
        print(f"Chunks created: {stats['chunks_created']}")
        print(f"Vectors indexed: {stats['vectors_indexed']}")
        print(f"Processing time: {stats['processing_time_ms']:.2f}ms")
        print(
            f"Average latency: {stats['performance_metrics']['avg_latency_ms']:.2f}ms"
        )
        print(f"Results saved to: {results_file}")

        return stats

    except Exception as e:
        logger.exception(f"Wikipedia ingestion failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
