"""
Project report writer for RAG system knowledge updates and project documentation.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
from datetime import datetime
import json
from .temperature_writer import TemperatureBasedWriter
from rag_system.core.pipeline import EnhancedRAGPipeline
from rag_system.core.cognitive_nexus import CognitiveNexus

logger = logging.getLogger(__name__)

class ProjectReportWriter(TemperatureBasedWriter):
    """
    Project report writer that:
    1. Documents project outcomes for RAG system
    2. Tracks project evolution and learning
    3. Generates structured knowledge
    4. Uses temperature-based refinement
    5. Includes metrics and performance data
    """
    
    def __init__(self, rag_pipeline: EnhancedRAGPipeline, cognitive_nexus: CognitiveNexus):
        super().__init__()
        self.rag_pipeline = rag_pipeline
        self.cognitive_nexus = cognitive_nexus
        self.report_types = {
            "project_completion": {
                "sections": [
                    "Executive Summary",
                    "Project Overview",
                    "Objectives and Achievements",
                    "Technical Implementation",
                    "Challenges and Solutions",
                    "Performance Metrics",
                    "Lessons Learned",
                    "Future Implications"
                ],
                "required_metrics": [
                    "completion_rate",
                    "performance_indicators",
                    "resource_utilization",
                    "quality_metrics"
                ]
            },
            "knowledge_acquisition": {
                "sections": [
                    "Knowledge Summary",
                    "New Concepts",
                    "Relationship Mappings",
                    "Integration Points",
                    "Application Areas",
                    "Knowledge Gaps",
                    "Future Learning Paths"
                ],
                "required_metrics": [
                    "knowledge_coverage",
                    "concept_relationships",
                    "integration_success",
                    "learning_efficiency"
                ]
            },
            "system_evolution": {
                "sections": [
                    "System State Overview",
                    "Architectural Changes",
                    "Capability Enhancements",
                    "Performance Evolution",
                    "Integration Updates",
                    "System Health",
                    "Future Roadmap"
                ],
                "required_metrics": [
                    "system_performance",
                    "capability_metrics",
                    "health_indicators",
                    "evolution_trajectory"
                ]
            }
        }

    async def generate_report(
        self,
        project_id: str,
        report_type: str,
        metrics: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a project report with temperature-based refinement."""
        # Get project data
        project_data = await self._gather_project_data(project_id)
        
        # Analyze project evolution
        evolution_analysis = await self._analyze_project_evolution(project_id)
        
        # Prepare context
        context = {
            "project_id": project_id,
            "report_type": report_type,
            "project_data": project_data,
            "evolution_analysis": evolution_analysis,
            "metrics": metrics,
            "template": self.report_types[report_type],
            **kwargs
        }
        
        # Validate metrics
        self._validate_metrics(metrics, report_type)
        
        # Generate report sections
        sections = await self._generate_sections(context)
        
        # Generate executive summary
        summary = await self._generate_summary(context, sections)
        
        # Generate knowledge insights
        insights = await self._generate_insights(context, sections)
        
        # Compile full report
        report = await self._compile_report(
            project_id,
            summary,
            sections,
            insights,
            context
        )
        
        # Update RAG system
        await self._update_knowledge_base(report)
        
        return report

    async def _gather_project_data(self, project_id: str) -> Dict[str, Any]:
        """Gather all relevant project data."""
        # Get project history from RAG system
        history = await self.rag_pipeline.get_project_history(project_id)
        
        # Get cognitive context
        cognitive_context = await self.cognitive_nexus.get_project_context(project_id)
        
        # Get performance data
        performance_data = await self.rag_pipeline.get_project_metrics(project_id)
        
        return {
            "history": history,
            "cognitive_context": cognitive_context,
            "performance_data": performance_data
        }

    async def _analyze_project_evolution(self, project_id: str) -> Dict[str, Any]:
        """Analyze how the project has evolved."""
        # Get evolution trajectory
        trajectory = await self.cognitive_nexus.analyze_evolution(project_id)
        
        # Analyze knowledge growth
        knowledge_growth = await self.rag_pipeline.analyze_knowledge_growth(project_id)
        
        # Analyze system improvements
        system_improvements = await self.cognitive_nexus.analyze_improvements(project_id)
        
        return {
            "trajectory": trajectory,
            "knowledge_growth": knowledge_growth,
            "system_improvements": system_improvements
        }

    def _validate_metrics(self, metrics: Dict[str, Any], report_type: str):
        """Validate that all required metrics are present."""
        required_metrics = self.report_types[report_type]["required_metrics"]
        missing_metrics = [m for m in required_metrics if m not in metrics]
        
        if missing_metrics:
            raise ValueError(f"Missing required metrics: {missing_metrics}")

    async def _generate_sections(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Generate report sections with temperature-based refinement."""
        sections = {}
        template = context["template"]
        
        for section in template["sections"]:
            prompt = self._create_section_prompt(section, context)
            result = await self.generate_with_temperatures(prompt, context)
            sections[section] = result["final_content"]
        
        return sections

    async def _generate_summary(
        self,
        context: Dict[str, Any],
        sections: Dict[str, str]
    ) -> Dict[str, Any]:
        """Generate executive summary with temperature-based refinement."""
        summary_prompt = self._create_summary_prompt(context, sections)
        result = await self.generate_with_temperatures(summary_prompt, context)
        
        return {
            "content": result["final_content"],
            "key_points": await self._extract_key_points(result["final_content"]),
            "metrics_summary": self._summarize_metrics(context["metrics"])
        }

    async def _generate_insights(
        self,
        context: Dict[str, Any],
        sections: Dict[str, str]
    ) -> Dict[str, Any]:
        """Generate knowledge insights from the report."""
        # Extract concepts and relationships
        concepts = await self.cognitive_nexus.extract_concepts(sections)
        relationships = await self.cognitive_nexus.analyze_relationships(concepts)
        
        # Generate insights
        insights_prompt = self._create_insights_prompt(context, concepts, relationships)
        result = await self.generate_with_temperatures(insights_prompt, context)
        
        return {
            "content": result["final_content"],
            "concepts": concepts,
            "relationships": relationships,
            "knowledge_graph_updates": await self._generate_graph_updates(concepts, relationships)
        }

    def _create_section_prompt(self, section: str, context: Dict[str, Any]) -> str:
        """Create prompt for section generation."""
        return f"""
        Generate the '{section}' section for a {context['report_type']} project report.
        
        Project Context:
        {json.dumps(context['project_data'], indent=2)}
        
        Evolution Analysis:
        {json.dumps(context['evolution_analysis'], indent=2)}
        
        Metrics:
        {json.dumps(context['metrics'], indent=2)}
        
        Requirements:
        1. Comprehensive coverage of {section}
        2. Data-driven analysis
        3. Clear insights and implications
        4. Actionable recommendations
        5. Future considerations
        
        Generate the section content.
        """

    def _create_summary_prompt(self, context: Dict[str, Any], sections: Dict[str, str]) -> str:
        """Create prompt for summary generation."""
        return f"""
        Generate an executive summary for the {context['report_type']} project report.
        
        Project ID: {context['project_id']}
        
        Key Metrics:
        {json.dumps(context['metrics'], indent=2)}
        
        Evolution Highlights:
        {json.dumps(context['evolution_analysis'], indent=2)}
        
        Requirements:
        1. High-level project overview
        2. Key achievements and insights
        3. Critical metrics and trends
        4. Major implications
        5. Future directions
        
        Generate the executive summary.
        """

    def _create_insights_prompt(
        self,
        context: Dict[str, Any],
        concepts: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> str:
        """Create prompt for insights generation."""
        return f"""
        Generate knowledge insights for the project report.
        
        Concepts:
        {json.dumps(concepts, indent=2)}
        
        Relationships:
        {json.dumps(relationships, indent=2)}
        
        Project Evolution:
        {json.dumps(context['evolution_analysis'], indent=2)}
        
        Requirements:
        1. Key knowledge gains
        2. Novel relationships discovered
        3. Integration opportunities
        4. Knowledge gaps identified
        5. Learning recommendations
        
        Generate the insights content.
        """

    async def _compile_report(
        self,
        project_id: str,
        summary: Dict[str, Any],
        sections: Dict[str, str],
        insights: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile full report content."""
        report = {
            "project_id": project_id,
            "report_type": context["report_type"],
            "timestamp": datetime.now().isoformat(),
            "summary": summary,
            "sections": sections,
            "insights": insights,
            "metrics": context["metrics"],
            "evolution_analysis": context["evolution_analysis"],
            "metadata": {
                "version": "1.0",
                "generator": "ProjectReportWriter",
                "template_used": context["template"]
            }
        }
        
        return report

    async def _update_knowledge_base(self, report: Dict[str, Any]):
        """Update RAG system with report knowledge."""
        # Add report document
        document = {
            "content": json.dumps(report, indent=2),
            "metadata": {
                "document_type": "project_report",
                "project_id": report["project_id"],
                "report_type": report["report_type"],
                "timestamp": report["timestamp"]
            }
        }
        await self.rag_pipeline.add_document(document)
        
        # Update knowledge graph
        await self.cognitive_nexus.update_knowledge_graph(
            report["insights"]["knowledge_graph_updates"]
        )
        
        # Update project metrics
        await self.rag_pipeline.update_project_metrics(
            report["project_id"],
            report["metrics"]
        )
        
        logger.info(f"Updated knowledge base with project report: {report['project_id']}")

    async def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content."""
        extraction_prompt = f"""
        Extract key points from this content:
        {content}
        
        Return a list of concise, important points.
        """
        
        result = await self.llms["analytical"].complete(extraction_prompt)
        return [point.strip() for point in result.text.split('\n') if point.strip()]

    def _summarize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of metrics with trends and highlights."""
        summary = {
            "highlights": [],
            "concerns": [],
            "trends": {}
        }
        
        for metric, value in metrics.items():
            if isinstance(value, dict) and "trend" in value:
                summary["trends"][metric] = value["trend"]
                if value.get("is_highlight"):
                    summary["highlights"].append(metric)
                if value.get("is_concern"):
                    summary["concerns"].append(metric)
        
        return summary

    async def _generate_graph_updates(
        self,
        concepts: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate knowledge graph updates."""
        return {
            "new_nodes": [
                {
                    "id": concept["id"],
                    "type": concept["type"],
                    "properties": concept["properties"]
                }
                for concept in concepts
            ],
            "new_edges": [
                {
                    "source": rel["source"],
                    "target": rel["target"],
                    "type": rel["type"],
                    "properties": rel["properties"]
                }
                for rel in relationships
            ]
        }

# Example usage:
if __name__ == "__main__":
    async def main():
        rag_pipeline = EnhancedRAGPipeline()  # Configure as needed
        cognitive_nexus = CognitiveNexus()  # Configure as needed
        
        writer = ProjectReportWriter(rag_pipeline, cognitive_nexus)
        report = await writer.generate_report(
            project_id="project123",
            report_type="project_completion",
            metrics={
                "completion_rate": 0.95,
                "performance_indicators": {
                    "accuracy": 0.92,
                    "latency": "120ms",
                    "trend": "improving"
                },
                "resource_utilization": {
                    "cpu": 0.75,
                    "memory": 0.68,
                    "trend": "stable"
                },
                "quality_metrics": {
                    "code_coverage": 0.88,
                    "bug_density": 0.02,
                    "trend": "improving"
                }
            }
        )
        
        print(json.dumps(report, indent=2))

    asyncio.run(main())
