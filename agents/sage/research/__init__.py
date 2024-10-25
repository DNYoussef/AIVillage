"""Research capabilities for the Sage agent."""

from .capabilities.capabilities import ResearchCapabilities
from ..web_scraping.scraper import WebScraper
from ..online_search.search_engine import OnlineSearchEngine
from ..report_generation.report_writer import ReportWriter

__all__ = [
    "ResearchCapabilities",
    "WebScraper",
    "OnlineSearchEngine",
    "ReportWriter"
]
