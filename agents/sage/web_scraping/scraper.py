"""
Dynamic web scraper with vision model integration for intelligent content extraction.
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import json  # Added missing import

try:
    from playwright.async_api import async_playwright, Browser, Page
except ImportError:
    logger.warning("playwright not installed. Please install with: pip install playwright && playwright install")
    async_playwright = None
    Browser = None
    Page = None

from playwright.async_api import async_playwright, Browser, Page

from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification
import torch
from PIL import Image
import io
from bs4 import BeautifulSoup
from transformers import LayoutLMv3Processor, LayoutLMv3ForSequenceClassification, pipeline
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np

from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.error_handling.error_handler import error_handler
from langroid.language_models.openai_gpt import OpenAIGPTConfig

logger = logging.getLogger(__name__)

@dataclass
class ScrapingResult:
    """Container for scraping results."""
    url: str
    content: Dict[str, Any]
    screenshot: Optional[bytes]
    timestamp: str
    metadata: Dict[str, Any]

class DynamicWebScraper:
    """
    Vision-enhanced web scraper that can:
    1. Take screenshots of websites
    2. Use vision models to analyze page structure
    3. Dynamically adjust scraping strategies
    4. Feed data into RAG system
    """
    
    def __init__(self, vision_model_config: Dict[str, Any], rag_pipeline: EnhancedRAGPipeline):
        self.vision_model_config = vision_model_config
        self.rag_pipeline = rag_pipeline
        self.llm = OpenAIGPTConfig(chat_model="gpt-4-vision-preview").create()
        self.browser: Optional[Browser] = None
        self.active_pages: List[Page] = []
        
        # Initialize LayoutLMv3 for visual analysis
        self.layout_processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base")
        self.layout_model = LayoutLMv3ForSequenceClassification.from_pretrained("microsoft/layoutlmv3-base")
        
        # Initialize content classification
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.vectorizer = TfidfVectorizer()
        
        # Initialize adaptive scraping
        self.scraping_patterns = self._load_scraping_patterns()
        self.success_metrics: Dict[str, float] = {}

    async def scrape_url(
        self,
        url: str,
        scraping_config: Optional[Dict[str, Any]] = None
    ) -> ScrapingResult:
        """
        Enhanced scraping with visual analysis and dynamic content handling.
        """
        if not self.browser:
            await self.initialize()

        try:
            # Create enhanced page
            page = await self._create_enhanced_page()
            self.active_pages.append(page)

            # Navigate with advanced options
            await self._enhanced_navigation(page, url)
            
            # Take screenshot for visual analysis
            screenshot_bytes = await self._capture_screenshot(page)
            
            # Perform visual layout analysis
            layout_analysis = await self._analyze_layout(screenshot_bytes, page)
            
            # Handle dynamic content
            dynamic_content = await self._handle_dynamic_content(page)
            
            # Generate adaptive scraping strategy
            strategy = await self._generate_adaptive_strategy(
                layout_analysis,
                dynamic_content
            )
            
            # Execute strategy with content classification
            content = await self._execute_classified_scraping(page, strategy)
            
            # Update success metrics
            await self._update_success_metrics(url, content)
            
            result = ScrapingResult(
                url=url,
                content=content,
                screenshot=screenshot_bytes,
                timestamp=datetime.now().isoformat(),
                metadata={
                    "layout_analysis": layout_analysis,
                    "dynamic_content": dynamic_content,
                    "strategy": strategy
                }
            )
            
            await self._update_rag_system(result)
            return result
            
        finally:
            if page in self.active_pages:
                self.active_pages.remove(page)
                await page.close()

    async def _create_enhanced_page(self) -> Page:
        """Create a page with enhanced capabilities."""
        page = await self.browser.new_page()
        
        # Add custom JavaScript functions
        await page.add_init_script("""
            window.scrollPageToBottom = async () => {
                await new Promise(resolve => {
                    let totalHeight = 0;
                    let distance = 100;
                    let timer = setInterval(() => {
                        let scrollHeight = document.body.scrollHeight;
                        window.scrollBy(0, distance);
                        totalHeight += distance;
                        
                        if(totalHeight >= scrollHeight){
                            clearInterval(timer);
                            resolve();
                        }
                    }, 100);
                });
            };

            window.getVisibleElements = () => {
                const elements = document.querySelectorAll('*');
                return Array.from(elements).filter(el => {
                    const rect = el.getBoundingClientRect();
                    return (
                        rect.top >= 0 &&
                        rect.left >= 0 &&
                        rect.bottom <= window.innerHeight &&
                        rect.right <= window.innerWidth
                    );
                });
            };

            window.observeDynamicChanges = () => {
                const changes = [];
                const observer = new MutationObserver(mutations => {
                    mutations.forEach(mutation => {
                        changes.push({
                            type: mutation.type,
                            target: mutation.target.outerHTML,
                            addedNodes: Array.from(mutation.addedNodes).map(n => n.outerHTML),
                            removedNodes: Array.from(mutation.removedNodes).map(n => n.outerHTML)
                        });
                    });
                });
                
                observer.observe(document.body, {
                    childList: true,
                    subtree: true,
                    attributes: true,
                    characterData: true
                });
                
                return changes;
            };
        """)
        
        return page

    async def _enhanced_navigation(self, page: Page, url: str):
        """Enhanced page navigation with dynamic content handling."""
        # Navigate with extended timeout and network idle
        await page.goto(
            url,
            wait_until="networkidle",
            timeout=60000
        )
        
        # Handle cookie banners and popups
        await self._handle_obstacles(page)
        
        # Scroll to load lazy content
        await page.evaluate("window.scrollPageToBottom()")
        await page.wait_for_timeout(2000)
        
        # Start observing dynamic changes
        await page.evaluate("window.observeDynamicChanges()")

    async def _analyze_layout(
        self,
        screenshot: bytes,
        page: Page
    ) -> Dict[str, Any]:
        """
        Analyze page layout using LayoutLMv3 and visual features.
        """
        # Convert screenshot to image
        image = Image.open(io.BytesIO(screenshot))
        
        # Get page content
        content = await page.content()
        soup = BeautifulSoup(content, 'html.parser')
        
        # Process with LayoutLMv3
        inputs = self.layout_processor(
            images=image,
            text=soup.get_text(),
            return_tensors="pt"
        )
        
        outputs = self.layout_model(**inputs)
        
        # Get visible elements
        visible_elements = await page.evaluate("window.getVisibleElements()")
        
        # Combine layout analysis
        layout_analysis = {
            "model_outputs": outputs.logits.detach().numpy().tolist(),
            "visible_elements": visible_elements,
            "content_areas": await self._identify_content_areas(page),
            "visual_hierarchy": await self._analyze_visual_hierarchy(page)
        }
        
        return layout_analysis

    async def _handle_dynamic_content(self, page: Page) -> Dict[str, Any]:
        """
        Enhanced dynamic content handling.
        """
        dynamic_content = {
            "mutations": [],
            "ajax_requests": [],
            "lazy_loaded": []
        }
        
        # Monitor network requests
        async def handle_request(request):
            if request.resource_type in ["xhr", "fetch"]:
                dynamic_content["ajax_requests"].append({
                    "url": request.url,
                    "method": request.method,
                    "timestamp": datetime.now().isoformat()
                })
        
        page.on("request", handle_request)
        
        # Get mutation records
        mutations = await page.evaluate("window.observeDynamicChanges()")
        dynamic_content["mutations"] = mutations
        
        # Handle infinite scroll
        await self._handle_infinite_scroll(page)
        
        # Handle lazy loading
        lazy_loaded = await self._handle_lazy_loading(page)
        dynamic_content["lazy_loaded"] = lazy_loaded
        
        return dynamic_content

    async def _generate_adaptive_strategy(
        self,
        layout_analysis: Dict[str, Any],
        dynamic_content: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate adaptive scraping strategy based on analysis.
        """
        # Analyze successful patterns
        successful_patterns = self._analyze_success_patterns()
        
        # Generate base strategy
        strategy = await self._generate_scraping_strategy(layout_analysis)
        
        # Adapt strategy based on dynamic content
        strategy = self._adapt_to_dynamic_content(strategy, dynamic_content)
        
        # Incorporate successful patterns
        strategy = self._incorporate_successful_patterns(
            strategy,
            successful_patterns
        )
        
        return strategy

    async def _execute_classified_scraping(
        self,
        page: Page,
        strategy: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute scraping with content classification.
        """
        content = {}
        
        for element in strategy["elements"]:
            try:
                # Extract content
                extracted = await self._extract_element_content(
                    page,
                    element
                )
                
                if extracted:
                    # Classify content
                    classification = await self._classify_content(extracted)
                    
                    # Prioritize and structure content
                    content[element["name"]] = {
                        "content": extracted,
                        "classification": classification,
                        "priority": await self._calculate_priority(
                            extracted,
                            classification,
                            element
                        )
                    }
            except Exception as e:
                logger.error(f"Error extracting {element['name']}: {str(e)}")
        
        return content

    async def _classify_content(self, content: str) -> Dict[str, Any]:
        """
        Classify content type and importance.
        """
        # Define classification labels
        labels = [
            "main_content",
            "navigation",
            "metadata",
            "advertisement",
            "user_generated"
        ]
        
        # Perform zero-shot classification
        result = await self.classifier(
            content,
            labels,
            multi_label=True
        )
        
        # Calculate content importance
        importance = await self._calculate_content_importance(
            content,
            result["scores"]
        )
        
        return {
            "labels": dict(zip(result["labels"], result["scores"])),
            "importance": importance
        }

    async def _calculate_priority(
        self,
        content: str,
        classification: Dict[str, Any],
        element: Dict[str, Any]
    ) -> float:
        """
        Calculate content priority based on multiple factors.
        """
        # Base priority from classification
        base_priority = classification["importance"]
        
        # Adjust based on element type
        type_weights = {
            "heading": 1.2,
            "main_content": 1.0,
            "navigation": 0.6,
            "metadata": 0.4
        }
        type_weight = type_weights.get(element.get("type", "other"), 0.5)
        
        # Adjust based on visual prominence
        visual_weight = element.get("visual_prominence", 0.5)
        
        # Calculate final priority
        priority = (
            base_priority * 0.4 +
            type_weight * 0.3 +
            visual_weight * 0.3
        )
        
        return min(1.0, max(0.0, priority))

    async def _handle_infinite_scroll(self, page: Page):
        """Handle infinite scroll content."""
        last_height = await page.evaluate("document.body.scrollHeight")
        
        while True:
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await page.wait_for_timeout(2000)
            
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    async def _handle_lazy_loading(self, page: Page) -> List[Dict[str, Any]]:
        """Handle lazy-loaded content."""
        lazy_loaded = []
        
        # Find lazy-loaded images
        images = await page.query_selector_all("img[loading='lazy']")
        for img in images:
            await self._ensure_element_loaded(img)
            lazy_loaded.append({
                "type": "image",
                "src": await img.get_attribute("src")
            })
        
        # Find lazy-loaded iframes
        iframes = await page.query_selector_all("iframe[loading='lazy']")
        for iframe in iframes:
            await self._ensure_element_loaded(iframe)
            lazy_loaded.append({
                "type": "iframe",
                "src": await iframe.get_attribute("src")
            })
        
        return lazy_loaded

    async def _ensure_element_loaded(self, element):
        """Ensure lazy-loaded element is loaded."""
        try:
            await element.scroll_into_view_if_needed()
            await element.wait_for_element_state("visible")
        except Exception as e:
            logger.error(f"Error ensuring element loaded: {str(e)}")

    def _analyze_success_patterns(self) -> Dict[str, Any]:
        """Analyze successful scraping patterns."""
        patterns = {}
        
        for url, success_rate in self.success_metrics.items():
            if success_rate > 0.8:  # Consider patterns from successful scrapes
                pattern = self.scraping_patterns.get(url, {})
                for selector, stats in pattern.items():
                    if selector not in patterns:
                        patterns[selector] = {"success_count": 0, "total": 0}
                    patterns[selector]["success_count"] += stats["success"]
                    patterns[selector]["total"] += 1
        
        return {
            selector: stats["success_count"] / stats["total"]
            for selector, stats in patterns.items()
            if stats["total"] > 0
        }

    async def _update_success_metrics(self, url: str, content: Dict[str, Any]):
        """Update scraping success metrics."""
        # Calculate success rate based on content quality
        success_rate = len([
            item for item in content.values()
            if item and item.get("priority", 0) > 0.5
        ]) / len(content) if content else 0
        
        # Update metrics
        self.success_metrics[url] = success_rate
        
        # Update scraping patterns
        if url not in self.scraping_patterns:
            self.scraping_patterns[url] = {}
        
        for name, item in content.items():
            if "selector" in item:
                if item["selector"] not in self.scraping_patterns[url]:
                    self.scraping_patterns[url][item["selector"]] = {
                        "success": 0,
                        "total": 0
                    }
                
                pattern = self.scraping_patterns[url][item["selector"]]
                pattern["total"] += 1
                if item.get("priority", 0) > 0.5:
                    pattern["success"] += 1

    def _load_scraping_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load initial scraping patterns."""
        return {
            "article": {
                "selectors": [
                    "article",
                    '[role="article"]',
                    ".post-content",
                    ".article-content"
                ],
                "score": 1.0
            },
            "main_content": {
                "selectors": [
                    "main",
                    "#main",
                    ".main-content",
                    '[role="main"]'
                ],
                "score": 0.9
            },
            "navigation": {
                "selectors": [
                    "nav",
                    "header",
                    "footer",
                    '[role="navigation"]'
                ],
                "score": 0.3
            }
        }

    async def __aenter__(self):
        """Initialize browser for context manager."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Cleanup browser for context manager."""
        await self.cleanup()

    @error_handler.handle_error
    async def initialize(self):
        """Initialize the browser and vision model."""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(headless=True)
        logger.info("Browser initialized")

    @error_handler.handle_error
    async def cleanup(self):
        """Clean up resources."""
        for page in self.active_pages:
            await page.close()
        if self.browser:
            await self.browser.close()
        logger.info("Browser closed and resources cleaned up")

    @error_handler.handle_error
    async def scrape_url(self, url: str, scraping_config: Optional[Dict[str, Any]] = None) -> ScrapingResult:
        """
        Scrape a URL using vision-guided dynamic scraping.
        """
        if not self.browser:
            await self.initialize()

        try:
            # Create new page
            page = await self.browser.new_page()
            self.active_pages.append(page)

            # Navigate to URL
            await page.goto(url, wait_until="networkidle")
            
            # Take screenshot
            screenshot_bytes = await self._capture_screenshot(page)
            
            # Analyze page structure using vision model
            structure_analysis = await self._analyze_page_structure(screenshot_bytes)
            
            # Generate dynamic scraping strategy
            scraping_strategy = await self._generate_scraping_strategy(structure_analysis)
            
            # Execute scraping strategy
            content = await self._execute_scraping_strategy(page, scraping_strategy)
            
            # Prepare result
            result = ScrapingResult(
                url=url,
                content=content,
                screenshot=screenshot_bytes,
                timestamp=datetime.now().isoformat(),
                metadata={
                    "structure_analysis": structure_analysis,
                    "scraping_strategy": scraping_strategy
                }
            )
            
            # Feed to RAG system
            await self._update_rag_system(result)
            
            return result
            
        finally:
            # Cleanup
            if page in self.active_pages:
                self.active_pages.remove(page)
                await page.close()

    async def _capture_screenshot(self, page: Page) -> bytes:
        """Capture full page screenshot."""
        screenshot = await page.screenshot(full_page=True)
        return screenshot

    async def _analyze_page_structure(self, screenshot: bytes) -> Dict[str, Any]:
        """
        Analyze page structure using vision model.
        """
        # Convert screenshot to base64
        image = Image.open(io.BytesIO(screenshot))
        
        # Prepare vision model prompt
        prompt = """Analyze this webpage screenshot and identify:
        1. Main content areas and their locations
        2. Navigation elements
        3. Important UI components
        4. Content hierarchy
        5. Key information blocks
        
        Provide a structured analysis that can be used for dynamic scraping."""
        
        # Get vision model analysis
        response = await self.llm.complete_vision(prompt, image)
        
        # Parse and structure the analysis
        return self._parse_vision_analysis(response.text)

    async def _generate_scraping_strategy(self, structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate dynamic scraping strategy based on vision analysis.
        """
        strategy_prompt = f"""
        Based on this page structure analysis:
        {json.dumps(structure_analysis, indent=2)}
        
        Generate a detailed scraping strategy that includes:
        1. Priority elements to scrape
        2. CSS selectors or XPath expressions
        3. Content extraction rules
        4. Special handling instructions
        
        Return a structured strategy object."""
        
        response = await self.llm.complete(strategy_prompt)
        return json.loads(response.text)

    async def _execute_scraping_strategy(self, page: Page, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the dynamic scraping strategy.
        """
        content = {}
        
        for element in strategy["elements"]:
            selector = element["selector"]
            extraction_type = element["type"]
            
            try:
                if extraction_type == "text":
                    content[element["name"]] = await page.text_content(selector)
                elif extraction_type == "attribute":
                    content[element["name"]] = await page.get_attribute(selector, element["attribute"])
                elif extraction_type == "list":
                    elements = await page.query_selector_all(selector)
                    content[element["name"]] = [
                        await elem.text_content() for elem in elements
                    ]
            except Exception as e:
                logger.error(f"Error extracting {element['name']}: {str(e)}")
                content[element["name"]] = None
        
        return content

    async def _update_rag_system(self, result: ScrapingResult):
        """
        Update RAG system with scraped content.
        """
        # Prepare document
        document = {
            "content": json.dumps(result.content),
            "metadata": {
                "source_url": result.url,
                "timestamp": result.timestamp,
                "scraping_metadata": result.metadata
            }
        }
        
        # Add to RAG system
        await self.rag_pipeline.add_document(document)
        logger.info(f"Updated RAG system with content from {result.url}")

    def _parse_vision_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """
        Parse vision model's text analysis into structured format.
        """
        # Use LLM to structure the analysis
        structuring_prompt = f"""
        Convert this vision analysis into a structured JSON format:
        {analysis_text}
        
        Include:
        1. Main content areas with coordinates
        2. Navigation elements
        3. UI components
        4. Content hierarchy
        5. Key information blocks
        
        Return valid JSON only."""
        
        response = self.llm.complete(structuring_prompt)
        return json.loads(response.text)

    @error_handler.handle_error
    async def batch_scrape(self, urls: List[str], max_concurrent: int = 5) -> List[ScrapingResult]:
        """
        Scrape multiple URLs concurrently.
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def scrape_with_semaphore(url: str) -> ScrapingResult:
            async with semaphore:
                return await self.scrape_url(url)
        
        tasks = [scrape_with_semaphore(url) for url in urls]
        return await asyncio.gather(*tasks)

# Example usage:
if __name__ == "__main__":
    async def main():
        vision_config = {
            "model": "gpt-4-vision-preview",
            "max_tokens": 4096
        }
        
        rag_pipeline = EnhancedRAGPipeline()  # Configure as needed
        
        async with DynamicWebScraper(vision_config, rag_pipeline) as scraper:
            result = await scraper.scrape_url("https://example.com")
            print(f"Scraped content: {result.content}")

    asyncio.run(main())

