"""
Temperature-based writing mixin that can be applied to any report writer.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import logging
from datetime import datetime
import json
from langroid.language_models.openai_gpt import OpenAIGPTConfig

logger = logging.getLogger(__name__)

@dataclass
class TemperatureBasedVersion:
    """Container for a temperature-based version of content."""
    content: str
    temperature: float
    metadata: Dict[str, Any]
    timestamp: str

class TemperatureBasedWriter:
    """
    Mixin that adds temperature-based writing capabilities to any report writer.
    
    Process:
    1. Initial writing at temp=1.0 (creative)
    2. Second version at temp=0.5 (balanced)
    3. Critique at temp=0.01 (analytical)
    4. Integration of versions
    5. Final critique at temp=0.5
    """
    
    def __init__(self):
        self.llm_configs = {
            "creative": OpenAIGPTConfig(chat_model="gpt-4", temperature=1.0),
            "balanced": OpenAIGPTConfig(chat_model="gpt-4", temperature=0.5),
            "analytical": OpenAIGPTConfig(chat_model="gpt-4", temperature=0.01)
        }
        self.llms = {
            key: config.create() for key, config in self.llm_configs.items()
        }

    async def generate_with_temperatures(
        self,
        prompt: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate content using the temperature-based approach.
        """
        # Generate initial versions
        creative_version = await self._generate_version(prompt, context, "creative")
        balanced_version = await self._generate_version(prompt, context, "balanced")
        
        # Generate critique
        critique = await self._generate_critique(
            prompt,
            [creative_version, balanced_version],
            "analytical"
        )
        
        # Integrate versions
        integrated_version = await self._integrate_versions(
            prompt,
            [creative_version, balanced_version],
            critique
        )
        
        # Final critique and refinement
        final_critique = await self._generate_critique(
            prompt,
            [integrated_version],
            "balanced"
        )
        
        # Final refinement
        final_version = await self._refine_content(
            integrated_version,
            final_critique,
            "balanced"
        )
        
        return {
            "final_content": final_version.content,
            "versions": [creative_version, balanced_version, integrated_version, final_version],
            "critiques": [critique, final_critique],
            "metadata": {
                "prompt": prompt,
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
        }

    async def _generate_version(
        self,
        prompt: str,
        context: Dict[str, Any],
        style: str
    ) -> TemperatureBasedVersion:
        """Generate a version using specified temperature."""
        enhanced_prompt = self._enhance_prompt(prompt, context, style)
        response = await self.llms[style].complete(enhanced_prompt)
        
        return TemperatureBasedVersion(
            content=response.text,
            temperature=self.llm_configs[style].temperature,
            metadata={
                "style": style,
                "prompt": enhanced_prompt
            },
            timestamp=datetime.now().isoformat()
        )

    async def _generate_critique(
        self,
        prompt: str,
        versions: List[TemperatureBasedVersion],
        style: str
    ) -> Dict[str, Any]:
        """Generate a critique of the versions."""
        critique_prompt = f"""
        Analyze these versions of content for prompt: '{prompt}'
        
        {self._format_versions(versions)}
        
        Provide a detailed critique addressing:
        1. Accuracy and factual correctness
        2. Clarity and coherence
        3. Depth of analysis
        4. Style and tone appropriateness
        5. Areas for improvement
        6. Strengths to maintain
        
        Return a structured critique with specific recommendations.
        """
        
        response = await self.llms[style].complete(critique_prompt)
        
        return {
            "content": response.text,
            "style": style,
            "timestamp": datetime.now().isoformat()
        }

    async def _integrate_versions(
        self,
        prompt: str,
        versions: List[TemperatureBasedVersion],
        critique: Dict[str, Any]
    ) -> TemperatureBasedVersion:
        """Integrate multiple versions considering the critique."""
        integration_prompt = f"""
        Integrate these versions for prompt: '{prompt}'
        
        {self._format_versions(versions)}
        
        Consider this critique:
        {critique['content']}
        
        Create a unified version that:
        1. Combines the strengths of each version
        2. Addresses the critique's points
        3. Maintains consistency and flow
        4. Optimizes for clarity and accuracy
        
        Return the integrated content.
        """
        
        response = await self.llms["balanced"].complete(integration_prompt)
        
        return TemperatureBasedVersion(
            content=response.text,
            temperature=0.5,
            metadata={
                "style": "integrated",
                "source_versions": [v.metadata for v in versions],
                "critique": critique
            },
            timestamp=datetime.now().isoformat()
        )

    async def _refine_content(
        self,
        version: TemperatureBasedVersion,
        critique: Dict[str, Any],
        style: str
    ) -> TemperatureBasedVersion:
        """Refine content based on critique."""
        refinement_prompt = f"""
        Refine this content:
        
        {version.content}
        
        Based on this critique:
        {critique['content']}
        
        Provide a refined version that:
        1. Addresses all critique points
        2. Maintains the content's strengths
        3. Improves clarity and coherence
        4. Ensures accuracy and completeness
        
        Return the refined content.
        """
        
        response = await self.llms[style].complete(refinement_prompt)
        
        return TemperatureBasedVersion(
            content=response.text,
            temperature=self.llm_configs[style].temperature,
            metadata={
                "style": "refined",
                "original_version": version.metadata,
                "critique": critique
            },
            timestamp=datetime.now().isoformat()
        )

    def _enhance_prompt(self, prompt: str, context: Dict[str, Any], style: str) -> str:
        """Enhance prompt based on style."""
        style_guides = {
            "creative": "Be creative and exploratory, focusing on novel insights and connections.",
            "balanced": "Be balanced and measured, focusing on clear, well-supported points.",
            "analytical": "Be analytical and precise, focusing on accuracy and logical analysis."
        }
        
        return f"""
        {prompt}
        
        Context:
        {json.dumps(context, indent=2)}
        
        Style Guide:
        {style_guides[style]}
        
        Requirements:
        1. Clear structure
        2. Evidence-based arguments
        3. Relevant examples
        4. Actionable insights
        5. {"Creative solutions" if style == "creative" else "Balanced recommendations" if style == "balanced" else "Precise analysis"}
        
        Return well-structured content.
        """

    def _format_versions(self, versions: List[TemperatureBasedVersion]) -> str:
        """Format versions for prompt inclusion."""
        formatted = []
        for i, version in enumerate(versions, 1):
            formatted.append(f"""
            Version {i} (Temperature: {version.temperature}):
            {version.content}
            """)
        return "\n".join(formatted)
