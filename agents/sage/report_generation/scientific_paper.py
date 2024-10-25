"""
Scientific paper writer with LaTeX support and temperature-based refinement.
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime
from .temperature_writer import TemperatureBasedWriter
from rag_system.core.pipeline import EnhancedRAGPipeline

logger = logging.getLogger(__name__)

class ScientificPaperWriter(TemperatureBasedWriter):
    """
    Scientific paper writer that:
    1. Uses LaTeX templates
    2. Manages citations and references
    3. Integrates data visualization
    4. Applies temperature-based refinement
    """
    
    def __init__(self, rag_pipeline: EnhancedRAGPipeline):
        super().__init__()
        self.rag_pipeline = rag_pipeline
        self.latex_templates = {
            "standard": self._load_template("standard"),
            "conference": self._load_template("conference"),
            "journal": self._load_template("journal")
        }

    async def generate_paper(
        self,
        title: str,
        abstract: str,
        sections: List[str],
        data: Dict[str, Any],
        template_type: str = "standard",
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a scientific paper with temperature-based refinement."""
        # Prepare context
        context = {
            "title": title,
            "abstract": abstract,
            "sections": sections,
            "data": data,
            "template": self.latex_templates[template_type],
            **kwargs
        }
        
        # Generate each section with temperature-based refinement
        section_contents = {}
        for section in sections:
            prompt = self._create_section_prompt(section, context)
            result = await self.generate_with_temperatures(prompt, context)
            section_contents[section] = result["final_content"]
        
        # Generate abstract with temperature-based refinement
        abstract_prompt = self._create_abstract_prompt(context)
        abstract_result = await self.generate_with_temperatures(abstract_prompt, context)
        
        # Compile full paper
        paper_content = await self._compile_paper(
            title,
            abstract_result["final_content"],
            section_contents,
            context
        )
        
        # Add citations and references
        paper_with_citations = await self._add_citations(paper_content, context)
        
        # Generate figures and tables
        paper_with_visuals = await self._add_visualizations(paper_with_citations, data)
        
        return {
            "content": paper_with_visuals,
            "metadata": {
                "title": title,
                "sections": sections,
                "template_type": template_type,
                "timestamp": datetime.now().isoformat()
            }
        }

    def _load_template(self, template_type: str) -> str:
        """Load LaTeX template."""
        # Example template structure
        return f"""
\\documentclass[12pt]{{article}}
\\usepackage{{amsmath}}
\\usepackage{{graphicx}}
\\usepackage{{hyperref}}
\\usepackage{{natbib}}

\\title{{%TITLE%}}
\\author{{%AUTHOR%}}
\\date{{\\today}}

\\begin{{document}}
\\maketitle

\\begin{{abstract}}
%ABSTRACT%
\\end{{abstract}}

%CONTENT%

\\bibliographystyle{{plain}}
\\bibliography{{references}}

\\end{{document}}
"""

    def _create_section_prompt(self, section: str, context: Dict[str, Any]) -> str:
        """Create prompt for section generation."""
        return f"""
        Write the '{section}' section for a scientific paper titled '{context["title"]}'.
        
        Paper Context:
        - Abstract: {context["abstract"]}
        - Available Data: {list(context["data"].keys())}
        
        Requirements:
        1. Academic writing style
        2. Clear logical flow
        3. Data-driven arguments
        4. Technical precision
        5. LaTeX compatibility
        
        Generate the section content.
        """

    def _create_abstract_prompt(self, context: Dict[str, Any]) -> str:
        """Create prompt for abstract generation."""
        return f"""
        Write an abstract for a scientific paper titled '{context["title"]}'.
        
        Paper Sections:
        {context["sections"]}
        
        Available Data:
        {list(context["data"].keys())}
        
        Requirements:
        1. Clear problem statement
        2. Methodology summary
        3. Key findings
        4. Main conclusions
        5. 250 words or less
        
        Generate the abstract.
        """

    async def _compile_paper(
        self,
        title: str,
        abstract: str,
        section_contents: Dict[str, str],
        context: Dict[str, Any]
    ) -> str:
        """Compile full paper content."""
        template = context["template"]
        content = []
        
        for section, text in section_contents.items():
            content.append(f"\\section{{{section}}}\n{text}")
        
        paper = template.replace("%TITLE%", title)
        paper = paper.replace("%ABSTRACT%", abstract)
        paper = paper.replace("%CONTENT%", "\n\n".join(content))
        
        return paper

    async def _add_citations(self, content: str, context: Dict[str, Any]) -> str:
        """Add citations and references."""
        # Query RAG system for relevant citations
        citations = await self.rag_pipeline.find_citations(content)
        
        # Add citations to content
        cited_content = content
        for citation in citations:
            cited_content = self._insert_citation(cited_content, citation)
        
        # Generate bibliography
        bib_content = self._generate_bibliography(citations)
        
        return f"{cited_content}\n\n{bib_content}"

    async def _add_visualizations(self, content: str, data: Dict[str, Any]) -> str:
        """Add figures and tables."""
        # Generate figures from data
        figures = await self._generate_figures(data)
        
        # Generate tables from data
        tables = await self._generate_tables(data)
        
        # Insert figures and tables into content
        enhanced_content = content
        for fig in figures:
            enhanced_content = self._insert_figure(enhanced_content, fig)
        for table in tables:
            enhanced_content = self._insert_table(enhanced_content, table)
        
        return enhanced_content

    def _insert_citation(self, content: str, citation: Dict[str, Any]) -> str:
        """Insert citation into content."""
        return content.replace(
            citation["text"],
            f"{citation['text']} \\cite{{{citation['key']}}}"
        )

    def _generate_bibliography(self, citations: List[Dict[str, Any]]) -> str:
        """Generate bibliography in BibTeX format."""
        bibtex_entries = []
        for citation in citations:
            entry = f"""
            @{citation['type']}{{{citation['key']},
                title = {{{citation['title']}}},
                author = {{{citation['authors']}}},
                year = {{{citation['year']}}},
                journal = {{{citation['journal']}}},
                volume = {{{citation['volume']}}},
                number = {{{citation['number']}}},
                pages = {{{citation['pages']}}}
            }}
            """
            bibtex_entries.append(entry)
        
        return "\n".join(bibtex_entries)

    async def _generate_figures(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate figures from data."""
        figures = []
        for key, values in data.items():
            if isinstance(values, (list, dict)):
                figure = {
                    "label": f"fig:{key}",
                    "caption": f"Figure showing {key}",
                    "content": self._create_figure_latex(key, values)
                }
                figures.append(figure)
        return figures

    async def _generate_tables(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tables from data."""
        tables = []
        for key, values in data.items():
            if isinstance(values, (list, dict)):
                table = {
                    "label": f"tab:{key}",
                    "caption": f"Table showing {key}",
                    "content": self._create_table_latex(key, values)
                }
                tables.append(table)
        return tables

    def _create_figure_latex(self, key: str, data: Any) -> str:
        """Create LaTeX figure environment."""
        return f"""
        \\begin{{figure}}[htbp]
            \\centering
            \\includegraphics[width=0.8\\textwidth]{{{key}}}
            \\caption{{Visualization of {key}}}
            \\label{{fig:{key}}}
        \\end{{figure}}
        """

    def _create_table_latex(self, key: str, data: Any) -> str:
        """Create LaTeX table environment."""
        return f"""
        \\begin{{table}}[htbp]
            \\centering
            \\caption{{Data for {key}}}
            \\label{{tab:{key}}}
            \\begin{{tabular}}{{|c|c|}}
                \\hline
                {self._format_table_data(data)}
                \\hline
            \\end{{tabular}}
        \\end{{table}}
        """

    def _format_table_data(self, data: Any) -> str:
        """Format data for LaTeX table."""
        if isinstance(data, dict):
            rows = [f"{k} & {v} \\\\" for k, v in data.items()]
            return "\n                ".join(rows)
        elif isinstance(data, list):
            rows = [f"{i} & {v} \\\\" for i, v in enumerate(data)]
            return "\n                ".join(rows)
        return ""

# Example usage:
if __name__ == "__main__":
    async def main():
        rag_pipeline = EnhancedRAGPipeline()  # Configure as needed
        
        writer = ScientificPaperWriter(rag_pipeline)
        paper = await writer.generate_paper(
            title="Advanced Multi-Agent Systems in AI Research",
            abstract="This paper explores...",
            sections=["Introduction", "Methodology", "Results", "Discussion"],
            data={
                "performance_metrics": {"accuracy": 0.95, "recall": 0.92},
                "system_architecture": "path/to/diagram.png"
            }
        )
        
        print(paper["content"])

    asyncio.run(main())
