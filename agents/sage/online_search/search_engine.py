"""
Enhanced search engine with semantic analysis, cross-validation, and multi-provider support.
"""

from typing import Dict, Any, List, Optional, Tuple
import asyncio
import logging
from datetime import datetime
import json
from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from transformers import (
    AutoTokenizer,
    AutoModel,
    pipeline,
    T5ForSequenceClassification
)
from sentence_transformers import SentenceTransformer
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except ImportError:
    logger.warning("spacy not installed. Please install with: pip install spacy && python -m spacy download en_core_web_sm")
    nlp = None

from urllib.parse import quote_plus, urlparse
import aiohttp
from abc import ABC, abstractmethod

from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.error_handling.error_handler import error_handler
from langroid.language_models.openai_gpt import OpenAIGPTConfig

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Container for search results."""
    query: str
    url: str
    title: str
    snippet: str
    source: str
    timestamp: str
    metadata: Dict[str, Any]

@dataclass
class RankedResult:
    """Container for a ranked search result."""
    result: SearchResult
    relevance_score: float
    trust_score: float
    usefulness_score: float
    combined_score: float
    source_quality: Dict[str, float]
    semantic_similarity: float
    temporal_score: float
    validation_score: float

class SearchProvider(ABC):
    """Base class for search providers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def initialize(self):
        """Initialize the search provider."""
        self.session = aiohttp.ClientSession()

    async def cleanup(self):
        """Clean up resources."""
        if self.session:
            await self.session.close()

    @abstractmethod
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """Perform search using the provider."""
        pass

class DuckDuckGoProvider(SearchProvider):
    """DuckDuckGo search provider (API-free)."""
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        if not self.session:
            await self.initialize()

        url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json"
        
        try:
            async with self.session.get(url) as response:
                data = await response.json()
                
                results = []
                for result in data.get("Results", []):
                    results.append(SearchResult(
                        query=query,
                        url=result.get("FirstURL", ""),
                        title=result.get("Text", ""),
                        snippet=result.get("Abstract", ""),
                        source="duckduckgo",
                        timestamp=datetime.now().isoformat(),
                        metadata={"raw_result": result}
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {str(e)}")
            return []

class GoogleCustomSearchProvider(SearchProvider):
    """Google Custom Search API provider."""
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        if not self.session:
            await self.initialize()

        api_key = self.config.get("api_key")
        cx = self.config.get("cx")
        
        if not api_key or not cx:
            raise ValueError("Google Custom Search requires api_key and cx")

        url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cx}&q={quote_plus(query)}"
        
        try:
            async with self.session.get(url) as response:
                data = await response.json()
                
                results = []
                for item in data.get("items", []):
                    results.append(SearchResult(
                        query=query,
                        url=item.get("link", ""),
                        title=item.get("title", ""),
                        snippet=item.get("snippet", ""),
                        source="google",
                        timestamp=datetime.now().isoformat(),
                        metadata={"raw_result": item}
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"Google search error: {str(e)}")
            return []

class BingSearchProvider(SearchProvider):
    """Bing Web Search API provider."""
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        if not self.session:
            await self.initialize()

        api_key = self.config.get("api_key")
        if not api_key:
            raise ValueError("Bing Search requires api_key")

        url = "https://api.bing.microsoft.com/v7.0/search"
        headers = {"Ocp-Apim-Subscription-Key": api_key}
        params = {"q": query, "count": kwargs.get("count", 10)}
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                data = await response.json()
                
                results = []
                for item in data.get("webPages", {}).get("value", []):
                    results.append(SearchResult(
                        query=query,
                        url=item.get("url", ""),
                        title=item.get("name", ""),
                        snippet=item.get("snippet", ""),
                        source="bing",
                        timestamp=datetime.now().isoformat(),
                        metadata={"raw_result": item}
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"Bing search error: {str(e)}")
            return []

class QwantSearchProvider(SearchProvider):
    """Qwant search provider."""
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        if not self.session:
            await self.initialize()

        url = "https://api.qwant.com/v3/search/web"
        headers = {"User-Agent": "Mozilla/5.0"}
        params = {
            "q": query,
            "count": kwargs.get("count", 10),
            "locale": kwargs.get("locale", "en_US")
        }
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                data = await response.json()
                
                results = []
                for item in data.get("data", {}).get("result", {}).get("items", []):
                    results.append(SearchResult(
                        query=query,
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("description", ""),
                        source="qwant",
                        timestamp=datetime.now().isoformat(),
                        metadata={"raw_result": item}
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"Qwant search error: {str(e)}")
            return []

class BraveSearchProvider(SearchProvider):
    """Brave Search API provider."""
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        if not self.session:
            await self.initialize()

        api_key = self.config.get("api_key")
        if not api_key:
            raise ValueError("Brave Search requires api_key")

        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {"X-Subscription-Token": api_key}
        params = {
            "q": query,
            "count": kwargs.get("count", 10)
        }
        
        try:
            async with self.session.get(url, headers=headers, params=params) as response:
                data = await response.json()
                
                results = []
                for item in data.get("web", {}).get("results", []):
                    results.append(SearchResult(
                        query=query,
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("description", ""),
                        source="brave",
                        timestamp=datetime.now().isoformat(),
                        metadata={"raw_result": item}
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"Brave search error: {str(e)}")
            return []

class SerpAPIProvider(SearchProvider):
    """SerpAPI provider with free tier support."""
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        if not self.session:
            await self.initialize()

        api_key = self.config.get("api_key")
        if not api_key:
            raise ValueError("SerpAPI requires api_key")

        url = "https://serpapi.com/search"
        params = {
            "q": query,
            "api_key": api_key,
            "engine": kwargs.get("engine", "google"),
            "num": kwargs.get("count", 10)
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                results = []
                for item in data.get("organic_results", []):
                    results.append(SearchResult(
                        query=query,
                        url=item.get("link", ""),
                        title=item.get("title", ""),
                        snippet=item.get("snippet", ""),
                        source="serpapi",
                        timestamp=datetime.now().isoformat(),
                        metadata={"raw_result": item}
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"SerpAPI search error: {str(e)}")
            return []

class YandexSearchProvider(SearchProvider):
    """Yandex Search API provider."""
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        if not self.session:
            await self.initialize()

        api_key = self.config.get("api_key")
        user = self.config.get("user")
        if not api_key or not user:
            raise ValueError("Yandex Search requires api_key and user")

        url = "https://yandex.com/search/xml"
        params = {
            "query": query,
            "key": api_key,
            "user": user,
            "l10n": kwargs.get("locale", "en"),
            "maxpassages": kwargs.get("count", 10)
        }
        
        try:
            async with self.session.get(url, params=params) as response:
                data = await response.json()
                
                results = []
                for item in data.get("response", {}).get("results", []):
                    results.append(SearchResult(
                        query=query,
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("passage", ""),
                        source="yandex",
                        timestamp=datetime.now().isoformat(),
                        metadata={"raw_result": item}
                    ))
                
                return results
                
        except Exception as e:
            logger.error(f"Yandex search error: {str(e)}")
            return []

class EnhancedSearchEngine:
    """
    Enhanced search engine with:
    1. Multi-provider support
    2. Semantic analysis and clustering
    3. Cross-validation and fact-checking
    4. Trust and usefulness scoring
    5. Temporal relevance
    6. Query expansion and variations
    7. Entity extraction and linking
    8. Topic modeling and analysis
    """
    
    def __init__(self, rag_pipeline: EnhancedRAGPipeline, config: Dict[str, Any]):
        self.rag_pipeline = rag_pipeline
        self.config = config
        
        # Initialize providers
        self.providers: Dict[str, SearchProvider] = {}
        self._initialize_providers()
        
        # Initialize NLP models
        self.llm = OpenAIGPTConfig(chat_model="gpt-4").create()
        self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.model = T5ForSequenceClassification.from_pretrained("t5-base")
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize analysis tools
        self.lda = LatentDirichletAllocation(n_components=10, random_state=42)
        self.vectorizer = TfidfVectorizer(max_features=1000)
        self.ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        
        # Initialize metrics
        self.trust_metrics = self._initialize_trust_metrics()

    def _initialize_providers(self):
        """Initialize configured search providers."""
        provider_configs = self.config.get("providers", {})
        
        provider_mapping = {
            "duckduckgo": DuckDuckGoProvider,
            "google": GoogleCustomSearchProvider,
            "bing": BingSearchProvider,
            "qwant": QwantSearchProvider,
            "brave": BraveSearchProvider,
            "serpapi": SerpAPIProvider,
            "yandex": YandexSearchProvider
        }
        
        for name, config in provider_configs.items():
            if name in provider_mapping:
                self.providers[name] = provider_mapping[name](config)

    @error_handler.handle_error
    async def search(
        self,
        query: str,
        depth: int = 3,
        min_score: float = 0.5,
        provider: Optional[str] = None
    ) -> List[RankedResult]:
        """
        Perform enhanced search with semantic analysis and validation.
        """
        # Generate query variations
        variations = await self._generate_query_variations(query, depth)
        
        # Perform searches
        all_results = []
        for variation in variations:
            if provider:
                if provider in self.providers:
                    results = await self.providers[provider].search(
                        variation["query"]
                    )
                    all_results.extend(results)
            else:
                tasks = []
                for provider in self.providers.values():
                    tasks.append(provider.search(variation["query"]))
                
                results_list = await asyncio.gather(*tasks)
                for results in results_list:
                    all_results.extend(results)
        
        # Remove duplicates
        unique_results = self._remove_duplicates(all_results)
        
        # Extract entities and topics
        entities = await self._extract_entities(query, unique_results)
        topics = await self._extract_topics(unique_results)
        
        # Semantic analysis
        semantic_results = await self._semantic_analysis(query, unique_results)
        
        # Cross-validate information
        validated_results = await self._cross_validate_results(semantic_results)
        
        # Time-based analysis
        temporal_results = self._analyze_temporal_relevance(validated_results)
        
        # Final ranking with all factors
        final_results = await self._comprehensive_ranking(
            temporal_results,
            entities,
            topics,
            min_score
        )
        
        # Update trust metrics
        await self._update_trust_metrics(final_results)
        
        return final_results

    async def _generate_query_variations(
        self,
        query: str,
        depth: int
    ) -> List[Dict[str, Any]]:
        """Generate variations of the search query."""
        prompt = f"""
        Generate {depth} different ways to search for information about:
        {query}
        
        For each variation, provide:
        1. Reworded query
        2. Search perspective (e.g., technical, practical, theoretical)
        3. Expected information type
        
        Return a JSON array of objects with fields:
        - query: reworded query
        - perspective: search perspective
        - expected_info: expected information type
        """
        
        response = await self.llm.complete(prompt)
        variations = json.loads(response.text)
        
        # Add original query
        variations.append({
            "query": query,
            "perspective": "original",
            "expected_info": "direct match"
        })
        
        return variations

    async def _semantic_analysis(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[Dict[str, Any]]:
        """Perform semantic analysis of results."""
        # Encode query and results
        query_embedding = self.sentence_transformer.encode(query)
        
        analyzed_results = []
        for result in results:
            # Get result embedding
            content = f"{result.title} {result.snippet}"
            result_embedding = self.sentence_transformer.encode(content)
            
            # Calculate semantic similarity
            similarity = np.dot(query_embedding, result_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(result_embedding)
            )
            
            # Extract entities
            entities = self.ner_pipeline(content)
            
            analyzed_results.append({
                "result": result,
                "embedding": result_embedding,
                "semantic_similarity": float(similarity),
                "entities": entities
            })
        
        return analyzed_results

    async def _extract_topics(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Extract topics from search results."""
        # Prepare documents
        documents = [
            f"{r.title} {r.snippet}"
            for r in results
        ]
        
        # Create document-term matrix
        dtm = self.vectorizer.fit_transform(documents)
        
        # Extract topics
        topics = self.lda.fit_transform(dtm)
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Format topics
        formatted_topics = []
        for topic_idx, topic in enumerate(self.lda.components_):
            top_words = [
                feature_names[i]
                for i in topic.argsort()[:-10:-1]
            ]
            formatted_topics.append({
                "id": topic_idx,
                "words": top_words,
                "weight": float(np.mean(topics[:, topic_idx]))
            })
        
        return formatted_topics

    async def _extract_entities(
        self,
        query: str,
        results: List[SearchResult]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Extract and analyze entities."""
        # Extract query entities
        query_doc = self.nlp(query)
        query_entities = [
            {"text": ent.text, "label": ent.label_}
            for ent in query_doc.ents
        ]
        
        # Extract result entities
        result_entities = {}
        for result in results:
            content = f"{result.title} {result.snippet}"
            doc = self.nlp(content)
            
            result_entities[result.url] = [
                {"text": ent.text, "label": ent.label_}
                for ent in doc.ents
            ]
        
        # Find entity overlaps
        entity_matches = self._find_entity_matches(
            query_entities,
            result_entities
        )
        
        return {
            "query_entities": query_entities,
            "result_entities": result_entities,
            "matches": entity_matches
        }

    def _find_entity_matches(
        self,
        query_entities: List[Dict[str, Any]],
        result_entities: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Find matching entities between query and results."""
        matches = {}
        for url, entities in result_entities.items():
            url_matches = []
            for q_ent in query_entities:
                for r_ent in entities:
                    if (
                        q_ent["text"].lower() in r_ent["text"].lower() or
                        r_ent["text"].lower() in q_ent["text"].lower()
                    ):
                        url_matches.append({
                            "query_entity": q_ent,
                            "result_entity": r_ent
                        })
            matches[url] = url_matches
        return matches

    def _analyze_temporal_relevance(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Analyze temporal relevance of results."""
        now = datetime.now()
        
        for result in results:
            # Extract timestamp if available
            timestamp = self._extract_timestamp(result["result"])
            
            if timestamp:
                age = now - timestamp
                # Calculate decay factor (1.0 to 0.0)
                temporal_score = max(0, min(1, 1 - (age.days / 365)))
            else:
                temporal_score = 0.5  # Default score for unknown age
            
            result["temporal_score"] = temporal_score
        
        return results

    def _extract_timestamp(self, result: SearchResult) -> Optional[datetime]:
        """Extract timestamp from result metadata."""
        try:
            if "timestamp" in result.metadata:
                return datetime.fromisoformat(result.metadata["timestamp"])
            elif "date" in result.metadata:
                return datetime.fromisoformat(result.metadata["date"])
            return None
        except:
            return None

    def _remove_duplicates(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate search results."""
        seen_urls = set()
        unique_results = []
        
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        return unique_results

    async def batch_search(
        self,
        queries: List[str],
        **kwargs
    ) -> Dict[str, List[RankedResult]]:
        """
        Perform batch search for multiple queries.
        """
        tasks = [self.search(query, **kwargs) for query in queries]
        results = await asyncio.gather(*tasks)
        return dict(zip(queries, results))

# Example usage:
if __name__ == "__main__":
    async def main():
        config = {
            "providers": {
                "duckduckgo": {},
                "google": {
                    "api_key": "your_api_key",
                    "cx": "your_search_engine_id"
                }
            }
        }
        
        rag_pipeline = EnhancedRAGPipeline()  # Configure as needed
        
        search_engine = EnhancedSearchEngine(rag_pipeline, config)
        results = await search_engine.search(
            query="quantum computing applications in cryptography",
            depth=3,
            min_score=0.6
        )
        
        for result in results:
            print(f"\nTitle: {result.result.title}")
            print(f"URL: {result.result.url}")
            print(f"Combined Score: {result.combined_score:.3f}")
            print(f"Trust Score: {result.trust_score:.3f}")
            print(f"Semantic Similarity: {result.semantic_similarity:.3f}")

    asyncio.run(main())
