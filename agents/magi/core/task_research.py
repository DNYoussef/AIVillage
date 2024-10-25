"""Task Research implementation."""

from typing import Dict, Any, List, Optional
import logging
import asyncio
from datetime import datetime

from ...utils.logging import get_logger
from ...utils.exceptions import AIVillageException
from ..research.integration import ResearchIntegration

logger = get_logger(__name__)

class TaskResearch:
    """
    Handles research tasks for the MAGI system.
    Integrates with various research sources and tools.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize TaskResearch with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.research_integration = ResearchIntegration(config)
        logger.info("TaskResearch initialized")

    async def research_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform research for a given task.
        
        Args:
            task: Task description and parameters
            
        Returns:
            Research results and findings
        """
        try:
            # Extract research requirements
            requirements = await self._extract_research_requirements(task)
            
            # Gather information from multiple sources
            results = await asyncio.gather(
                self._search_knowledge_base(requirements),
                self._search_external_sources(requirements),
                self._analyze_related_tasks(requirements)
            )
            
            # Combine and analyze results
            combined_results = await self._combine_research_results(results)
            analysis = await self._analyze_research_results(combined_results)
            
            return {
                'requirements': requirements,
                'results': combined_results,
                'analysis': analysis,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.exception(f"Error researching task: {str(e)}")
            raise AIVillageException(f"Error researching task: {str(e)}")

    async def _extract_research_requirements(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract research requirements from task.
        
        Args:
            task: Task description and parameters
            
        Returns:
            Research requirements
        """
        try:
            return await self.research_integration.extract_requirements(task)
        except Exception as e:
            logger.exception(f"Error extracting research requirements: {str(e)}")
            raise AIVillageException(f"Error extracting research requirements: {str(e)}")

    async def _search_knowledge_base(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search internal knowledge base.
        
        Args:
            requirements: Research requirements
            
        Returns:
            Knowledge base search results
        """
        try:
            return await self.research_integration.search_knowledge_base(requirements)
        except Exception as e:
            logger.exception(f"Error searching knowledge base: {str(e)}")
            raise AIVillageException(f"Error searching knowledge base: {str(e)}")

    async def _search_external_sources(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Search external information sources.
        
        Args:
            requirements: Research requirements
            
        Returns:
            External source search results
        """
        try:
            return await self.research_integration.search_external_sources(requirements)
        except Exception as e:
            logger.exception(f"Error searching external sources: {str(e)}")
            raise AIVillageException(f"Error searching external sources: {str(e)}")

    async def _analyze_related_tasks(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Analyze related tasks and their outcomes.
        
        Args:
            requirements: Research requirements
            
        Returns:
            Related task analysis results
        """
        try:
            return await self.research_integration.analyze_related_tasks(requirements)
        except Exception as e:
            logger.exception(f"Error analyzing related tasks: {str(e)}")
            raise AIVillageException(f"Error analyzing related tasks: {str(e)}")

    async def _combine_research_results(self, results: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Combine results from different sources.
        
        Args:
            results: List of results from different sources
            
        Returns:
            Combined research results
        """
        try:
            return await self.research_integration.combine_results(results)
        except Exception as e:
            logger.exception(f"Error combining research results: {str(e)}")
            raise AIVillageException(f"Error combining research results: {str(e)}")

    async def _analyze_research_results(self, combined_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze combined research results.
        
        Args:
            combined_results: Combined results from all sources
            
        Returns:
            Analysis of research results
        """
        try:
            return await self.research_integration.analyze_results(combined_results)
        except Exception as e:
            logger.exception(f"Error analyzing research results: {str(e)}")
            raise AIVillageException(f"Error analyzing research results: {str(e)}")

    async def get_research_status(self) -> Dict[str, Any]:
        """
        Get current research system status.
        
        Returns:
            Status information for the research system
        """
        try:
            return await self.research_integration.get_status()
        except Exception as e:
            logger.exception(f"Error getting research status: {str(e)}")
            raise AIVillageException(f"Error getting research status: {str(e)}")

    async def save_state(self, path: str):
        """
        Save research system state.
        
        Args:
            path: Path to save state
        """
        try:
            await self.research_integration.save_state(path)
            logger.info(f"Research system state saved to {path}")
        except Exception as e:
            logger.exception(f"Error saving research system state: {str(e)}")
            raise AIVillageException(f"Error saving research system state: {str(e)}")

    async def load_state(self, path: str):
        """
        Load research system state.
        
        Args:
            path: Path to load state from
        """
        try:
            await self.research_integration.load_state(path)
            logger.info(f"Research system state loaded from {path}")
        except Exception as e:
            logger.exception(f"Error loading research system state: {str(e)}")
            raise AIVillageException(f"Error loading research system state: {str(e)}")
