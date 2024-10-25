"""
Dynamic wiki-style content generator that adapts to user comprehension level.
"""

from typing import Dict, Any, List, Optional, Union
import asyncio
import logging
from datetime import datetime
import json
from .temperature_writer import TemperatureBasedWriter
from rag_system.core.pipeline import EnhancedRAGPipeline

logger = logging.getLogger(__name__)

class DynamicWikiWriter(TemperatureBasedWriter):
    """
    Wiki-style content generator that:
    1. Adapts to user comprehension level
    2. Generates Wikipedia-style articles
    3. Uses temperature-based refinement
    4. Includes cross-references and citations
    """
    
    def __init__(self, rag_pipeline: EnhancedRAGPipeline):
        super().__init__()
        self.rag_pipeline = rag_pipeline
        self.comprehension_levels = {
            "beginner": {
                "complexity": 0.3,
                "technical_depth": 0.2,
                "example_frequency": 0.8,
                "visualization_frequency": 0.7
            },
            "intermediate": {
                "complexity": 0.6,
                "technical_depth": 0.5,
                "example_frequency": 0.5,
                "visualization_frequency": 0.5
            },
            "advanced": {
                "complexity": 0.9,
                "technical_depth": 0.8,
                "example_frequency": 0.3,
                "visualization_frequency": 0.3
            },
            "expert": {
                "complexity": 1.0,
                "technical_depth": 1.0,
                "example_frequency": 0.2,
                "visualization_frequency": 0.2
            }
        }

    async def generate_article(
        self,
        topic: str,
        user_level: str,
        user_context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a wiki-style article adapted to user comprehension."""
        # Analyze user context and determine optimal level
        if user_context:
            optimal_level = await self._analyze_user_context(user_context)
            level_params = self._adjust_level_params(
                self.comprehension_levels[user_level],
                optimal_level
            )
        else:
            level_params = self.comprehension_levels[user_level]
        
        # Prepare context
        context = {
            "topic": topic,
            "level_params": level_params,
            "user_context": user_context,
            **kwargs
        }
        
        # Generate article sections
        sections = await self._generate_sections(context)
        
        # Generate summary with temperature-based refinement
        summary_prompt = self._create_summary_prompt(context, sections)
        summary_result = await self.generate_with_temperatures(summary_prompt, context)
        
        # Add cross-references and citations
        sections_with_refs = await self._add_references(sections)
        
        # Compile full article
        article_content = await self._compile_article(
            topic,
            summary_result["final_content"],
            sections_with_refs,
            context
        )
        
        # Add related topics and suggestions
        article_with_related = await self._add_related_content(article_content, context)
        
        # Update RAG system
        await self._update_rag_system(article_with_related)
        
        return article_with_related

    async def _analyze_user_context(self, user_context: Dict[str, Any]) -> Dict[str, float]:
        """Analyze user context to determine optimal comprehension level."""
        # Query RAG system for user interaction history
        history = await self.rag_pipeline.get_user_history(user_context.get("user_id"))
        
        # Analyze question complexity
        questions = history.get("questions", [])
        complexity_scores = await self._analyze_question_complexity(questions)
        
        # Analyze response understanding
        responses = history.get("responses", [])
        understanding_scores = await self._analyze_response_understanding(responses)
        
        # Calculate optimal parameters
        return {
            "complexity": sum(s["complexity"] for s in complexity_scores) / len(complexity_scores),
            "technical_depth": sum(s["technical_depth"] for s in understanding_scores) / len(understanding_scores),
            "example_frequency": 1.0 - (sum(s["abstraction"] for s in understanding_scores) / len(understanding_scores)),
            "visualization_frequency": sum(s["visual_preference"] for s in complexity_scores) / len(complexity_scores)
        }

    def _adjust_level_params(
        self,
        base_params: Dict[str, float],
        optimal_params: Dict[str, float]
    ) -> Dict[str, float]:
        """Adjust level parameters based on optimal parameters."""
        return {
            key: (base_params[key] + optimal_params[key]) / 2
            for key in base_params
        }

    async def _generate_sections(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Generate article sections with temperature-based refinement."""
        # Determine sections based on topic and level
        sections = await self._determine_sections(context)
        
        # Generate each section
        section_contents = {}
        for section in sections:
            prompt = self._create_section_prompt(section, context)
            result = await self.generate_with_temperatures(prompt, context)
            section_contents[section] = result["final_content"]
        
        return section_contents

    async def _determine_sections(self, context: Dict[str, Any]) -> List[str]:
        """Determine appropriate sections based on topic and level."""
        level_params = context["level_params"]
        
        # Base sections
        sections = ["Overview", "Main Concepts"]
        
        # Add sections based on complexity
        if level_params["complexity"] > 0.4:
            sections.extend(["Historical Context", "Applications"])
        
        if level_params["complexity"] > 0.6:
            sections.extend(["Technical Details", "Current Research"])
        
        if level_params["complexity"] > 0.8:
            sections.extend(["Advanced Topics", "Future Directions"])
        
        # Add examples section if needed
        if level_params["example_frequency"] > 0.5:
            sections.append("Examples and Use Cases")
        
        return sections

    def _create_section_prompt(self, section: str, context: Dict[str, Any]) -> str:
        """Create prompt for section generation."""
        level_params = context["level_params"]
        
        return f"""
        Generate the '{section}' section for a wiki-style article about '{context["topic"]}'.
        
        Comprehension Parameters:
        - Complexity: {level_params['complexity']}
        - Technical Depth: {level_params['technical_depth']}
        - Example Frequency: {level_params['example_frequency']}
        - Visualization Frequency: {level_params['visualization_frequency']}
        
        Requirements:
        1. Adapt complexity to user level
        2. Include appropriate technical details
        3. Use examples when helpful
        4. Suggest visualizations where relevant
        5. Maintain Wikipedia-style tone
        
        Generate the section content.
        """

    def _create_summary_prompt(self, context: Dict[str, Any], sections: Dict[str, str]) -> str:
        """Create prompt for summary generation."""
        level_params = context["level_params"]
        
        return f"""
        Generate a summary for a wiki-style article about '{context["topic"]}'.
        
        Comprehension Level:
        {json.dumps(level_params, indent=2)}
        
        Available Sections:
        {list(sections.keys())}
        
        Requirements:
        1. Clear introduction to the topic
        2. Highlight key concepts
        3. Preview main sections
        4. Adapt complexity to user level
        5. Engage reader's interest
        
        Generate the summary.
        """

    async def _add_references(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """Add cross-references and citations."""
        # Query RAG system for relevant references
        references = await self.rag_pipeline.find_references(sections)
        
        # Add citations to sections
        sections_with_refs = {}
        for section_name, content in sections.items():
            sections_with_refs[section_name] = {
                "content": self._insert_citations(content, references),
                "references": self._filter_references(references, content)
            }
        
        return sections_with_refs

    async def _compile_article(
        self,
        topic: str,
        summary: str,
        sections: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile full article content."""
        article = {
            "title": topic,
            "summary": summary,
            "sections": sections,
            "comprehension_level": {
                "base_level": context.get("user_level", "intermediate"),
                "adjusted_params": context["level_params"]
            },
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "version": "1.0",
                "generator": "DynamicWikiWriter"
            }
        }
        
        return article

    async def _add_related_content(
        self,
        article: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Add related topics and suggested readings."""
        # Find related topics in RAG system
        related = await self.rag_pipeline.find_related_topics(article["title"])
        
        # Adjust suggestions based on user level
        level_params = context["level_params"]
        filtered_related = self._filter_related_content(related, level_params)
        
        article["related_content"] = {
            "topics": filtered_related["topics"],
            "suggested_reading": filtered_related["reading"],
            "see_also": filtered_related["see_also"]
        }
        
        return article

    def _insert_citations(self, content: str, references: List[Dict[str, Any]]) -> str:
        """Insert citations into content."""
        cited_content = content
        for ref in references:
            cited_content = cited_content.replace(
                ref["text"],
                f"{ref['text']} [{ref['id']}]"
            )
        return cited_content

    def _filter_references(
        self,
        references: List[Dict[str, Any]],
        content: str
    ) -> List[Dict[str, Any]]:
        """Filter references relevant to specific content."""
        return [
            ref for ref in references
            if ref["text"] in content
        ]

    def _filter_related_content(
        self,
        related: Dict[str, Any],
        level_params: Dict[str, float]
    ) -> Dict[str, Any]:
        """Filter related content based on comprehension level."""
        filtered = {
            "topics": [],
            "reading": [],
            "see_also": []
        }
        
        for topic in related["topics"]:
            if topic["complexity"] <= level_params["complexity"]:
                filtered["topics"].append(topic)
        
        for reading in related["reading"]:
            if reading["technical_level"] <= level_params["technical_depth"]:
                filtered["reading"].append(reading)
        
        filtered["see_also"] = related["see_also"][:5]  # Limit to top 5
        
        return filtered

    async def _update_rag_system(self, article: Dict[str, Any]):
        """Update RAG system with the generated article."""
        document = {
            "content": json.dumps(article, indent=2),
            "metadata": {
                "document_type": "wiki_article",
                "topic": article["title"],
                "comprehension_level": article["comprehension_level"],
                "timestamp": article["metadata"]["generated_at"]
            }
        }
        
        await self.rag_pipeline.add_document(document)
        logger.info(f"Updated RAG system with wiki article: {article['title']}")

# Example usage:
if __name__ == "__main__":
    async def main():
        rag_pipeline = EnhancedRAGPipeline()  # Configure as needed
        
        writer = DynamicWikiWriter(rag_pipeline)
        article = await writer.generate_article(
            topic="Quantum Computing",
            user_level="intermediate",
            user_context={
                "user_id": "user123",
                "interests": ["physics", "computer science"],
                "background": "software engineering"
            }
        )
        
        print(json.dumps(article, indent=2))

    asyncio.run(main())
