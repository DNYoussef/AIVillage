"""
Agent report writer for inter-agent communication and RAG system updates.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
from datetime import datetime
import json
from .temperature_writer import TemperatureBasedWriter
from rag_system.core.pipeline import EnhancedRAGPipeline
from agents.unified_base_agent import UnifiedBaseAgent

logger = logging.getLogger(__name__)

class AgentReportWriter(TemperatureBasedWriter):
    """
    Report writer for agent-to-agent communication that:
    1. Structures information for agent consumption
    2. Integrates with RAG system
    3. Uses temperature-based refinement
    4. Includes metadata for processing
    """
    
    def __init__(self, rag_pipeline: EnhancedRAGPipeline):
        super().__init__()
        self.rag_pipeline = rag_pipeline
        self.report_templates = {
            "task_result": self._load_template("task_result"),
            "analysis": self._load_template("analysis"),
            "recommendation": self._load_template("recommendation"),
            "status_update": self._load_template("status_update")
        }

    async def generate_report(
        self,
        report_type: str,
        content: Dict[str, Any],
        source_agent: UnifiedBaseAgent,
        target_agent: Optional[UnifiedBaseAgent] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate an agent report with temperature-based refinement."""
        # Prepare context
        context = {
            "report_type": report_type,
            "content": content,
            "source_agent": source_agent.info,
            "target_agent": target_agent.info if target_agent else None,
            "template": self.report_templates[report_type],
            **kwargs
        }
        
        # Generate report sections
        sections = await self._generate_sections(context)
        
        # Generate summary with temperature-based refinement
        summary_prompt = self._create_summary_prompt(context, sections)
        summary_result = await self.generate_with_temperatures(summary_prompt, context)
        
        # Compile full report
        report_content = await self._compile_report(
            summary_result["final_content"],
            sections,
            context
        )
        
        # Add metadata
        report_with_metadata = self._add_metadata(report_content, context)
        
        # Update RAG system
        await self._update_rag_system(report_with_metadata)
        
        return report_with_metadata

    def _load_template(self, report_type: str) -> Dict[str, Any]:
        """Load report template."""
        templates = {
            "task_result": {
                "sections": [
                    "Task Overview",
                    "Methodology",
                    "Results",
                    "Implications",
                    "Next Steps"
                ],
                "metadata_schema": {
                    "task_id": str,
                    "completion_status": str,
                    "priority": int,
                    "dependencies": List[str]
                }
            },
            "analysis": {
                "sections": [
                    "Context",
                    "Key Findings",
                    "Detailed Analysis",
                    "Recommendations",
                    "Supporting Data"
                ],
                "metadata_schema": {
                    "analysis_type": str,
                    "confidence_level": float,
                    "data_sources": List[str]
                }
            },
            "recommendation": {
                "sections": [
                    "Executive Summary",
                    "Current State",
                    "Proposed Changes",
                    "Impact Analysis",
                    "Implementation Plan"
                ],
                "metadata_schema": {
                    "urgency": str,
                    "impact_level": str,
                    "required_resources": Dict[str, Any]
                }
            },
            "status_update": {
                "sections": [
                    "Current Status",
                    "Progress Updates",
                    "Blockers",
                    "Resource Utilization",
                    "Timeline Updates"
                ],
                "metadata_schema": {
                    "status": str,
                    "progress_percentage": float,
                    "health_indicators": Dict[str, str]
                }
            }
        }
        return templates.get(report_type, templates["task_result"])

    async def _generate_sections(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Generate each section with temperature-based refinement."""
        sections = {}
        template = context["template"]
        
        for section in template["sections"]:
            prompt = self._create_section_prompt(section, context)
            result = await self.generate_with_temperatures(prompt, context)
            sections[section] = result["final_content"]
        
        return sections

    def _create_section_prompt(self, section: str, context: Dict[str, Any]) -> str:
        """Create prompt for section generation."""
        return f"""
        Generate the '{section}' section for an agent report.
        
        Report Type: {context['report_type']}
        Source Agent: {context['source_agent']['name']}
        Target Agent: {context['target_agent']['name'] if context['target_agent'] else 'All Agents'}
        
        Content Context:
        {json.dumps(context['content'], indent=2)}
        
        Requirements:
        1. Clear and concise information
        2. Relevant to target agent's capabilities
        3. Actionable insights
        4. Structured data when applicable
        5. Machine-parseable format
        
        Generate the section content.
        """

    def _create_summary_prompt(self, context: Dict[str, Any], sections: Dict[str, str]) -> str:
        """Create prompt for summary generation."""
        return f"""
        Generate a summary for an agent report.
        
        Report Type: {context['report_type']}
        Source Agent: {context['source_agent']['name']}
        Target Agent: {context['target_agent']['name'] if context['target_agent'] else 'All Agents'}
        
        Sections:
        {json.dumps(sections, indent=2)}
        
        Requirements:
        1. High-level overview
        2. Key points from each section
        3. Critical insights
        4. Required actions
        5. Priority indicators
        
        Generate the summary.
        """

    async def _compile_report(
        self,
        summary: str,
        sections: Dict[str, str],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile full report content."""
        report = {
            "summary": summary,
            "sections": sections,
            "source_agent": context["source_agent"]["name"],
            "target_agent": context["target_agent"]["name"] if context["target_agent"] else "All Agents",
            "report_type": context["report_type"],
            "timestamp": datetime.now().isoformat()
        }
        
        return report

    def _add_metadata(self, report: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Add metadata for agent consumption."""
        template = context["template"]
        schema = template["metadata_schema"]
        
        metadata = {
            "schema_version": "1.0",
            "report_id": f"{context['report_type']}_{datetime.now().isoformat()}",
            "source_agent_type": context["source_agent"]["description"],
            "target_agent_type": context["target_agent"]["description"] if context["target_agent"] else "all",
            "processing_priority": context.get("priority", 1),
            "required_capabilities": context["source_agent"]["capabilities"],
            "schema": schema
        }
        
        report["metadata"] = metadata
        return report

    async def _update_rag_system(self, report: Dict[str, Any]):
        """Update RAG system with the report."""
        document = {
            "content": json.dumps(report, indent=2),
            "metadata": {
                "document_type": "agent_report",
                "report_type": report["report_type"],
                "source_agent": report["source_agent"],
                "target_agent": report["target_agent"],
                "timestamp": report["timestamp"]
            }
        }
        
        await self.rag_pipeline.add_document(document)
        logger.info(f"Updated RAG system with {report['report_type']} report from {report['source_agent']}")

# Example usage:
if __name__ == "__main__":
    async def main():
        rag_pipeline = EnhancedRAGPipeline()  # Configure as needed
        
        writer = AgentReportWriter(rag_pipeline)
        report = await writer.generate_report(
            report_type="task_result",
            content={
                "task": "Analyze system performance",
                "results": {"accuracy": 0.95, "latency": "120ms"},
                "recommendations": ["Scale up resources", "Optimize queries"]
            },
            source_agent=UnifiedBaseAgent(name="AnalysisAgent"),
            target_agent=UnifiedBaseAgent(name="SupervisorAgent")
        )
        
        print(json.dumps(report, indent=2))

    asyncio.run(main())
