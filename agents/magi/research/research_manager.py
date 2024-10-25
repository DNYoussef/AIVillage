"""Research manager for MAGI."""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from ..integrations.github_client import GitHubClient, GitHubRepo
from ..integrations.huggingface_client import HuggingFaceClient, HFModel

logger = logging.getLogger(__name__)

@dataclass
class ResearchResult:
    """Represents a research result."""
    query: str
    github_results: List[GitHubRepo]
    huggingface_results: List[HFModel]
    code_samples: List[Dict[str, Any]]
    analysis: str
    timestamp: datetime

class ResearchManager:
    """
    Manages code and model research using GitHub and Hugging Face.
    """
    
    def __init__(
        self,
        github_token: Optional[str] = None,
        huggingface_token: Optional[str] = None,
        llm: Any = None
    ):
        self.github_client = GitHubClient(github_token)
        self.huggingface_client = HuggingFaceClient(huggingface_token)
        self.llm = llm
        self.research_history: List[ResearchResult] = []
        self.downloads_dir = "downloads"
        os.makedirs(self.downloads_dir, exist_ok=True)

    async def research_implementation(
        self,
        query: str,
        language: Optional[str] = None,
        include_models: bool = True,
        max_results: int = 5
    ) -> ResearchResult:
        """
        Research implementation approaches using GitHub and optionally Hugging Face.

        :param query: Search query
        :param language: Programming language filter
        :param include_models: Whether to include model search
        :param max_results: Maximum results per source
        :return: Research results
        """
        # Search GitHub repositories
        github_results = await self.github_client.search_repositories(
            query=query,
            language=language,
            max_results=max_results
        )
        
        # Search code samples
        code_samples = await self.github_client.search_code(
            query=query,
            language=language,
            max_results=max_results
        )
        
        # Search Hugging Face models if requested
        huggingface_results = []
        if include_models:
            huggingface_results = await self.huggingface_client.search_models(
                query=query,
                limit=max_results
            )
            
        # Analyze results
        analysis = await self._analyze_results(
            query, github_results, huggingface_results, code_samples)
            
        result = ResearchResult(
            query=query,
            github_results=github_results,
            huggingface_results=huggingface_results,
            code_samples=code_samples,
            analysis=analysis,
            timestamp=datetime.now()
        )
        
        self.research_history.append(result)
        return result

    async def download_implementation(
        self,
        repo: GitHubRepo,
        local_dir: Optional[str] = None
    ) -> str:
        """
        Download a GitHub implementation.

        :param repo: GitHub repository to download
        :param local_dir: Optional local directory
        :return: Path to downloaded implementation
        """
        if local_dir is None:
            local_dir = os.path.join(self.downloads_dir, 'github', repo.name)
            
        return await self.github_client.clone_repository(
            repo_full_name=repo.full_name,
            local_path=local_dir
        )

    async def download_model(
        self,
        model: HFModel,
        local_dir: Optional[str] = None
    ) -> str:
        """
        Download a Hugging Face model.

        :param model: Hugging Face model to download
        :param local_dir: Optional local directory
        :return: Path to downloaded model
        """
        if local_dir is None:
            local_dir = os.path.join(self.downloads_dir, 'huggingface', model.name)
            
        return await self.huggingface_client.download_model(
            model_id=model.id,
            local_dir=local_dir
        )

    async def analyze_implementation(
        self,
        repo: GitHubRepo
    ) -> Dict[str, Any]:
        """
        Analyze a GitHub implementation.

        :param repo: GitHub repository to analyze
        :return: Analysis results
        """
        # Get code samples
        code_samples = await self.github_client.search_code(
            query=f"repo:{repo.full_name}",
            max_results=10
        )
        
        # Generate analysis prompt
        analysis_prompt = f"""
        Analyze this GitHub repository implementation:
        
        Repository: {repo.full_name}
        Description: {repo.description}
        Language: {repo.language}
        Stars: {repo.stars}
        
        Code samples:
        {code_samples}
        
        Provide:
        1. Key implementation approaches
        2. Notable patterns or techniques
        3. Potential improvements
        4. Integration considerations
        """
        
        analysis = await self.llm.generate(analysis_prompt)
        
        return {
            'repository': repo.full_name,
            'code_samples': code_samples,
            'analysis': analysis
        }

    async def analyze_model(self, model: HFModel) -> Dict[str, Any]:
        """
        Analyze a Hugging Face model.

        :param model: Hugging Face model to analyze
        :return: Analysis results
        """
        # Get model details and card
        details = await self.huggingface_client.get_model_details(model.id)
        model_card = await self.huggingface_client.get_model_card(model.id)
        
        # Generate analysis prompt
        analysis_prompt = f"""
        Analyze this Hugging Face model:
        
        Model: {model.id}
        Description: {model.description}
        Type: {model.model_type}
        Pipeline: {model.pipeline_tag}
        
        Model Card:
        {model_card}
        
        Provide:
        1. Key capabilities and use cases
        2. Performance characteristics
        3. Integration requirements
        4. Potential limitations
        """
        
        analysis = await self.llm.generate(analysis_prompt)
        
        return {
            'model': model.id,
            'details': details,
            'model_card': model_card,
            'analysis': analysis
        }

    async def _analyze_results(
        self,
        query: str,
        github_results: List[GitHubRepo],
        huggingface_results: List[HFModel],
        code_samples: List[Dict[str, Any]]
    ) -> str:
        """Generate comprehensive analysis of research results."""
        analysis_prompt = f"""
        Analyze these research results for query: {query}

        GitHub Repositories:
        {[repo.full_name for repo in github_results]}

        Hugging Face Models:
        {[model.id for model in huggingface_results]}

        Code Samples: {len(code_samples)} samples found

        Provide:
        1. Most relevant implementations and approaches
        2. Common patterns and techniques
        3. Recommended integration strategies
        4. Potential challenges and solutions
        5. Best practices identified
        """
        
        return await self.llm.generate(analysis_prompt)

    def get_research_history(
        self,
        query: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[ResearchResult]:
        """
        Get research history with optional filters.

        :param query: Filter by query
        :param start_date: Filter by start date
        :param end_date: Filter by end date
        :return: Filtered research history
        """
        results = self.research_history
        
        if query:
            results = [r for r in results if query.lower() in r.query.lower()]
            
        if start_date:
            results = [r for r in results if r.timestamp >= start_date]
            
        if end_date:
            results = [r for r in results if r.timestamp <= end_date]
            
        return results

    async def get_implementation_suggestions(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Get implementation suggestions based on research.

        :param query: Implementation query
        :param context: Additional context
        :return: Implementation suggestions
        """
        # Research implementations
        research_result = await self.research_implementation(query)
        
        # Generate suggestions prompt
        suggestions_prompt = f"""
        Generate implementation suggestions based on research:

        Query: {query}
        Context: {context}

        Research Results:
        {research_result.analysis}

        Provide:
        1. Recommended implementation approach
        2. Key components and patterns to use
        3. Integration strategy
        4. Potential challenges and mitigations
        5. Testing considerations
        """
        
        suggestions = await self.llm.generate(suggestions_prompt)
        
        return {
            'query': query,
            'context': context,
            'research_result': research_result,
            'suggestions': suggestions
        }
