"""
Report management system for Sage agent.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
from datetime import datetime
from .report_generation.temperature_writer import TemperatureBasedWriter
from .report_generation.scientific_paper import ScientificPaperWriter
from .report_generation.agent_report import AgentReportWriter
from .report_generation.dynamic_wiki import DynamicWikiWriter
from .report_generation.project_report import ProjectReportWriter
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.core.cognitive_nexus import CognitiveNexus

logger = logging.getLogger(__name__)

class ReportManager:
    """
    Unified report management system that:
    1. Coordinates different report writers
    2. Manages report generation and storage
    3. Integrates with RAG system
    4. Handles report versioning and updates
    """
    
    def __init__(self, rag_pipeline: EnhancedRAGPipeline, cognitive_nexus: CognitiveNexus):
        self.rag_pipeline = rag_pipeline
        self.cognitive_nexus = cognitive_nexus
        
        # Initialize report writers
        self.scientific_writer = ScientificPaperWriter(rag_pipeline)
        self.agent_writer = AgentReportWriter(rag_pipeline)
        self.wiki_writer = DynamicWikiWriter(rag_pipeline)
        self.project_writer = ProjectReportWriter(rag_pipeline, cognitive_nexus)
        
        self.report_history: Dict[str, List[Dict[str, Any]]] = {}

    async def generate_report(
        self,
        report_type: str,
        content: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a report using the appropriate writer."""
        writer_map = {
            "scientific": self.scientific_writer.generate_paper,
            "agent": self.agent_writer.generate_report,
            "wiki": self.wiki_writer.generate_article,
            "project": self.project_writer.generate_report
        }
        
        if report_type not in writer_map:
            raise ValueError(f"Unknown report type: {report_type}")
        
        report = await writer_map[report_type](**content, **kwargs)
        
        # Store in history
        report_id = f"{report_type}_{datetime.now().isoformat()}"
        if report_id not in self.report_history:
            self.report_history[report_id] = []
        self.report_history[report_id].append(report)
        
        return report

    async def update_report(
        self,
        report_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update an existing report."""
        if report_id not in self.report_history:
            raise ValueError(f"Report not found: {report_id}")
        
        # Get latest version
        latest_version = self.report_history[report_id][-1]
        
        # Merge updates
        updated_content = self._merge_updates(latest_version, updates)
        
        # Generate new version
        report_type = report_id.split("_")[0]
        new_version = await self.generate_report(report_type, updated_content)
        
        return new_version

    def _merge_updates(
        self,
        original: Dict[str, Any],
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge updates into original content."""
        merged = original.copy()
        for key, value in updates.items():
            if isinstance(value, dict) and key in merged:
                merged[key] = self._merge_updates(merged[key], value)
            else:
                merged[key] = value
        return merged

    async def get_report_history(self, report_id: str) -> List[Dict[str, Any]]:
        """Get the history of a report."""
        if report_id not in self.report_history:
            raise ValueError(f"Report not found: {report_id}")
        return self.report_history[report_id]

    async def search_reports(
        self,
        query: str,
        report_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search through existing reports."""
        results = []
        for report_id, versions in self.report_history.items():
            if report_type and not report_id.startswith(report_type):
                continue
            
            latest_version = versions[-1]
            if self._matches_query(latest_version, query):
                results.append({
                    "report_id": report_id,
                    "report": latest_version,
                    "version_count": len(versions)
                })
        
        return results

    def _matches_query(self, report: Dict[str, Any], query: str) -> bool:
        """Check if report matches search query."""
        query = query.lower()
        
        # Check title/summary
        if "title" in report and query in report["title"].lower():
            return True
        if "summary" in report and query in str(report["summary"]).lower():
            return True
        
        # Check content
        if "content" in report and query in str(report["content"]).lower():
            return True
        
        # Check sections
        if "sections" in report:
            for section in report["sections"].values():
                if query in str(section).lower():
                    return True
        
        return False
