"""Research integration for MagiAgent."""

import logging
from typing import Dict, Any, List, Optional
from ..research.research_manager import ResearchManager, ResearchResult

logger = logging.getLogger(__name__)

class ResearchIntegration:
    """
    Integrates research capabilities into MagiAgent.
    """
    
    def __init__(
        self,
        github_token: Optional[str] = None,
        huggingface_token: Optional[str] = None,
        llm: Any = None
    ):
        self.research_manager = ResearchManager(
            github_token=github_token,
            huggingface_token=huggingface_token,
            llm=llm
        )

    async def research_for_task(
        self,
        task: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform research for a task.

        :param task: Task description and requirements
        :return: Research results and recommendations
        """
        # Generate research queries
        queries = await self._generate_research_queries(task)
        
        # Perform research for each query
        results = []
        for query in queries:
            result = await self.research_manager.research_implementation(
                query=query,
                language=task.get('language'),
                include_models='model' in task.get('requirements', [])
            )
            results.append(result)
            
        # Analyze results for task
        analysis = await self._analyze_results_for_task(task, results)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(task, results, analysis)
        
        return {
            'task': task,
            'research_results': results,
            'analysis': analysis,
            'recommendations': recommendations
        }

    async def download_resources(
        self,
        research_result: ResearchResult,
        recommendations: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Download recommended resources.

        :param research_result: Research results
        :param recommendations: Implementation recommendations
        :return: Paths to downloaded resources
        """
        downloads = {}
        
        # Download recommended GitHub repositories
        for repo in research_result.github_results:
            if repo.full_name in recommendations.get('recommended_repos', []):
                path = await self.research_manager.download_implementation(repo)
                downloads[f"github/{repo.full_name}"] = path
                
        # Download recommended models
        for model in research_result.huggingface_results:
            if model.id in recommendations.get('recommended_models', []):
                path = await self.research_manager.download_model(model)
                downloads[f"huggingface/{model.id}"] = path
                
        return downloads

    async def analyze_implementation_approach(
        self,
        task: Dict[str, Any],
        research_results: List[ResearchResult]
    ) -> Dict[str, Any]:
        """
        Analyze implementation approaches from research.

        :param task: Task description and requirements
        :param research_results: Research results
        :return: Implementation analysis
        """
        # Collect all relevant implementations
        implementations = []
        for result in research_results:
            for repo in result.github_results:
                analysis = await self.research_manager.analyze_implementation(repo)
                implementations.append(analysis)
                
        # Analyze models if relevant
        models = []
        if 'model' in task.get('requirements', []):
            for result in research_results:
                for model in result.huggingface_results:
                    analysis = await self.research_manager.analyze_model(model)
                    models.append(analysis)
                    
        # Generate comprehensive analysis
        analysis_prompt = f"""
        Analyze implementation approaches for task:
        
        Task: {task}
        
        GitHub Implementations:
        {implementations}
        
        Models:
        {models}
        
        Provide:
        1. Most suitable implementation approaches
        2. Key patterns and techniques to use
        3. Integration strategies
        4. Potential challenges and solutions
        5. Best practices to follow
        """
        
        analysis = await self.research_manager.llm.generate(analysis_prompt)
        
        return {
            'task': task,
            'implementations': implementations,
            'models': models,
            'analysis': analysis
        }

    async def _generate_research_queries(
        self,
        task: Dict[str, Any]
    ) -> List[str]:
        """Generate research queries from task."""
        query_prompt = f"""
        Generate specific research queries for this task:
        
        Task: {task}
        
        Generate:
        1. Implementation-focused queries
        2. Pattern-focused queries
        3. Integration-focused queries
        4. Model-focused queries (if relevant)
        
        Each query should be specific and targeted.
        """
        
        queries_text = await self.research_manager.llm.generate(query_prompt)
        return [q.strip() for q in queries_text.split('\n') if q.strip()]

    async def _analyze_results_for_task(
        self,
        task: Dict[str, Any],
        results: List[ResearchResult]
    ) -> str:
        """Analyze research results in context of task."""
        analysis_prompt = f"""
        Analyze these research results for task:
        
        Task: {task}
        
        Research Results:
        {[result.analysis for result in results]}
        
        Provide:
        1. Most relevant findings for this task
        2. Key implementation insights
        3. Recommended approaches
        4. Integration considerations
        5. Potential challenges and solutions
        """
        
        return await self.research_manager.llm.generate(analysis_prompt)

    async def _generate_recommendations(
        self,
        task: Dict[str, Any],
        results: List[ResearchResult],
        analysis: str
    ) -> Dict[str, Any]:
        """Generate implementation recommendations."""
        recommendations_prompt = f"""
        Generate implementation recommendations based on research:
        
        Task: {task}
        Analysis: {analysis}
        
        Provide:
        1. Recommended GitHub repositories to use
        2. Recommended models to use (if relevant)
        3. Key implementation patterns to follow
        4. Integration approach
        5. Testing strategy
        """
        
        recommendations_text = await self.research_manager.llm.generate(
            recommendations_prompt)
            
        # Parse recommendations
        recommended_repos = []
        recommended_models = []
        patterns = []
        integration_steps = []
        testing_steps = []
        
        current_section = None
        for line in recommendations_text.split('\n'):
            if 'repositories:' in line.lower():
                current_section = 'repos'
            elif 'models:' in line.lower():
                current_section = 'models'
            elif 'patterns:' in line.lower():
                current_section = 'patterns'
            elif 'integration:' in line.lower():
                current_section = 'integration'
            elif 'testing:' in line.lower():
                current_section = 'testing'
            elif line.strip():
                if current_section == 'repos':
                    recommended_repos.append(line.strip())
                elif current_section == 'models':
                    recommended_models.append(line.strip())
                elif current_section == 'patterns':
                    patterns.append(line.strip())
                elif current_section == 'integration':
                    integration_steps.append(line.strip())
                elif current_section == 'testing':
                    testing_steps.append(line.strip())
                    
        return {
            'recommended_repos': recommended_repos,
            'recommended_models': recommended_models,
            'patterns': patterns,
            'integration_steps': integration_steps,
            'testing_steps': testing_steps
        }
